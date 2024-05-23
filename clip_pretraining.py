import sys
import os
import torch
import numpy as np
import pickle
import shutil
import torch.nn as nn
import torch.nn.functional as F
import time

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

# torch.manual_seed(0)
# np.random.seed(0)

sys.path.append('.')
from models.clip_pretrainer import LLMPretraining, OFormerLLMPretraining, CLIPPretraining, OFormerCLIPPretraining
from models.clip_pretrainer import OFormerInputLLMPretraining

from utils import TransformerOperatorDataset2D, ElectricTransformerOperatorDataset2D
from anthony_data_handling import PDEDataset2D

import yaml
from tqdm import tqdm
import h5py
from matplotlib import pyplot as plt
import random


def get_model(embedding, config):
    if(embedding == 'llm'):
        model = LLMPretraining(llm=config['llm'], embed_dim=config['embed_dim'], im_size=config['im_size'],
                               initial_step=config['initial_step']+5 if(config['coeff']) else config['initial_step'])
    elif(embedding == 'oformerllm'):
        model = OFormerLLMPretraining(llm=config['llm'], embed_dim=config['embed_dim'], im_size=config['im_size'],
                                      initial_step=config['initial_step']+5 if(config['coeff']) else config['initial_step'])
    elif(embedding == 'oformerinputllm'):
        model = OFormerInputLLMPretraining(llm=config['llm'], embed_dim=config['embed_dim'], im_size=config['im_size'],
                                      initial_step=config['initial_step']+5 if(config['coeff']) else config['initial_step'])
    elif(embedding == 'clip'):
        model = CLIPPretraining(llm=config['llm'], embed_dim=config['embed_dim'], im_size=config['im_size'],
                                initial_step=config['initial_step']+5 if(config['coeff']) else config['initial_step'])
    
    elif(embedding == 'oformerclip'):
        model = OFormerCLIPPretraining(llm=config['llm'], embed_dim=config['embed_dim'], im_size=config['im_size'],
                                initial_step=config['initial_step']+5 if(config['coeff']) else config['initial_step'])
    model.to(device)
    return model


def new_get_data(f, config, pretraining=False):
    train_data = PDEDataset2D(
            path="/home/cooperlorsung/2d_heat_adv_burgers_train_large.h5",
            pde="Heat, Burgers, Advection",
            subset='heat,adv,burger',
            mode="train",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=False,
            device='cuda:0',
            clip='clip' in config['embedding'],
            spatial_downsampling=config['downsample'],
            llm=config['llm'],
            sentence='llm' in config['embedding'],
            num_samples=config['pretraining_num_samples'],
            temporal_horizon=config['sim_time'],
    )
    val_data = PDEDataset2D(
            path="/home/cooperlorsung/2d_heat_adv_burgers_valid_large.h5",
            pde="Heat, Burgers, Advection",
            subset='heat,adv,burger',
            mode="valid",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=True,
            device='cuda:0',
            clip='clip' in config['embedding'],
            spatial_downsampling=config['downsample'],
            llm=config['llm'],
            sentence='llm' in config['embedding'],
            temporal_horizon=config['sim_time'],
    )
    test_data = PDEDataset2D(
            path="/home/cooperlorsung/2d_heat_adv_burgers_test_large.h5",
            pde="Heat, Burgers, Advection",
            subset='heat,adv,burger',
            mode="test",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=True,
            device='cuda:0',
            clip='clip' in config['embedding'],
            spatial_downsampling=config['downsample'],
            llm=config['llm'],
            sentence='llm' in config['embedding'],
            temporal_horizon=config['sim_time'],
    )
    #if('electric' not in config['data_name']):
    #train_data.choose_subset('heat,adv,burger', n=config['pretraining_num_samples'])
    #val_data.choose_subset(config['data_name'])
    #test_data.choose_subset(config['data_name'])

    batch_size = config['pretraining_batch_size'] if(pretraining) else config['batch_size']
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, generator=torch.Generator(device='cpu'),
                                               num_workers=config['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, generator=torch.Generator(device='cpu'),
                                             num_workers=config['num_workers'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             num_workers=config['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def run_pretraining(model_name, config, prefix):
    #prefix = config['data_name'].split("_")[0]
    path = "./noise_weighted_pretrained_clip_modules/{}/{}_{}".format(config['pretraining_num_samples'], model_name, prefix)
    model_path = path + "/{}_{}_{}_{}.pt".format(config['seed'], config['llm'], config['pretraining_num_samples'], config['embedding'])
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')

    # Create the transformer model.
    transformer = get_model(config['embedding'], config)
    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    train_loader, val_loader, test_loader = new_get_data(f, config, pretraining=True)

    ################################################################
    # training and evaluation
    ################################################################

    if(config['return_text']):
        _data, _, _ = next(iter(val_loader))
        #_data, _, _, _ = next(iter(val_loader))
    else:
        _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)

    break_pretrain = config['pretraining_epochs'] == 0
    if(break_pretrain):
        config['pretraining_epochs'] += 1

    # Use Adam as the optimizer.
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['pretraining_learning_rate'],
                                 weight_decay=config['pretraining_weight_decay'])
    #    optimizer = torch.optim.Adam(list(list(model.parameters()) + list(conditioner.proj_up.parameters())),
    #                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    #else: # Only optimize base model
    #    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


    #TODO Make this step lr
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['pretraining_learning_rate'],
                                                    epochs=config['pretraining_epochs'], steps_per_epoch=len(train_loader))

    # Use mean squared error as the loss function.
    loss_fn = nn.L1Loss(reduction='mean')
    clip_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    # Try noise_weighted CLIP type loss function?
    loss_fn = nn.L1Loss(reduction='mean')

    # Train the transformer for the specified number of epochs.
    train_losses = []
    val_losses = []
    loss_val_min = np.infty
    #src_mask = generate_square_subsequent_mask(640).cuda()
    lrs = []
    shift = 0

    # Use this to select initial step and also coefficients
    mask = [False]*(config['sim_time']+5)
    mask[:config['initial_step']] = [True]*config['initial_step']
    mask[-5:] = [True]*5
    mask = torch.Tensor(mask).bool()

    print("\nPRETRAINNIG...")
    for epoch in tqdm(range(config['pretraining_epochs'])):
        if(break_pretrain):
            config['pretraining_epochs'] -= 1
            break
        # Iterate over the training dataset.
        train_loss = 0
        times = []
        max_val = 0
        transformer.train()
        #for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
        for bn, (x0, grid, sentence_embeddings) in enumerate(train_loader):
            start = time.time()
            if(config['train_style'] == 'next_step'):
                raise
            else:
                #print(x0.shape)
                y = x0[:, config['sim_time']-1,...,0].unsqueeze(-1)
                x0 = x0[:,mask].permute(0,2,3,1)
                grid = grid.permute(0,2,3,1)

            # Put data on correct device
            x0 = x0.to(device).float()
            y = y.to(device).float()
            grid = grid.to(device).float()

            # Add noise to x0...
            x0 += torch.randn(x0.shape).to(x0.device)*1.e-6

            # Calculate similarity matrix
            coeffs = x0[:,0,0,-5:]
            sim_mat = torch.sqrt(torch.sum((coeffs.unsqueeze(0) * coeffs.unsqueeze(1)).abs(), dim=-1))
            norm_vec = torch.max(torch.cat((coeffs.norm(dim=-1).unsqueeze(-1),
                                            coeffs.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
            norm_mat1 = torch.ones(coeffs.shape[0]).unsqueeze(0).to(norm_vec.device) * norm_vec.unsqueeze(1)
            norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(coeffs.shape[0]).unsqueeze(1).to(norm_vec.device)
            norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]
            sim_mat /= norm_mat


            # Add coefficient information
            #if(config['coeff']):
            #    nu = coeffs['nu'].unsqueeze(-1).to(device=x0.device)
            #    ax = coeffs['ax'].unsqueeze(-1).to(device=x0.device)
            #    ay = coeffs['ay'].unsqueeze(-1).to(device=x0.device)
            #    cx = coeffs['cx'].unsqueeze(-1).to(device=x0.device)
            #    cy = coeffs['cy'].unsqueeze(-1).to(device=x0.device)
            #    coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],     x0.shape[2], 5)
            #    inp = torch.cat((x0, coeff), dim=-1)
            #else: # Or don't
            #    inp = x0

            # Forward pass
            #print(type(x0), type(grid), type(sentence_embeddings))
            y_pred = transformer(x0, grid, sentence_embeddings, True)
            loss = loss_fn(y_pred, sim_mat)

            #labels = torch.arange(y_pred.shape[1]).to(device)
            #loss = clip_loss_fn(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            scheduler.step()

        train_loss /= (bn + 1)
        train_losses.append(train_loss)

        with torch.no_grad():
            transformer.eval()
            val_loss = 0
            all_val_preds = []
            #for bn, (x0, y, grid, tokens, t, sentence_embeddings) in enumerate(val_loader):
            #for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(val_loader):
            for bn, (x0, grid, sentence_embeddings) in enumerate(val_loader):
                # Forward pass: compute predictions by passing the input sequence
                # through the transformer.
                # Put data on correct device
                if(config['train_style'] == 'next_step'):
                    raise
                else:
                    y = x0[:, config['sim_time']-1,...,0].unsqueeze(-1)
                    x0 = x0[:,mask].permute(0,2,3,1)
                    grid = grid.permute(0,2,3,1)

                x0 = x0.to(device).float()
                y = y.to(device).float()
                grid = grid.to(device).float()

                coeffs = x0[:,0,0,-5:]
                sim_mat = torch.sqrt(torch.sum((coeffs.unsqueeze(0) * coeffs.unsqueeze(1)).abs(), dim=-1))
                norm_vec = torch.max(torch.cat((coeffs.norm(dim=-1).unsqueeze(-1),
                                                coeffs.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
                norm_mat1 = torch.ones(coeffs.shape[0]).unsqueeze(0).to(norm_vec.device) * norm_vec.unsqueeze(1)
                norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(coeffs.shape[0]).unsqueeze(1).to(norm_vec.device)
                norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]
                sim_mat /= norm_mat

                y_pred = transformer(x0, grid, sentence_embeddings, True)
                loss = loss_fn(y_pred, sim_mat)
                #labels = torch.arange(y_pred.shape[1]).to(device).repeat(y_pred.shape[0], 1)
                #labels = torch.arange(y_pred.shape[1]).to(device)
                #loss = clip_loss_fn(y_pred, labels)
                val_loss += loss.item()

            if  val_loss < loss_val_min:
                loss_val_min = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val_min
                    }, model_path)

        val_loss /= (bn + 1)
        val_losses.append(val_loss)

        # Print the loss at the end of each epoch.
        if(epoch%config['log_freq'] == 0):
            np.save("./{}/pretraining_train_l2s_{}.npy".format(path, seed), train_losses)
            np.save("./{}/pretraining_val_l2s_{}.npy".format(path, seed), val_losses)
            np.save("./{}/pretraining_lrs_{}.npy".format(path, seed), lrs)
            print(f"Epoch {epoch+1}: loss = {train_loss:.6f}\t val loss = {val_loss:.6f}")

    # Final update
    np.save("./{}/pretraining_train_l2s_{}.npy".format(path, seed), train_losses)
    np.save("./{}/pretraining_val_l2s_{}.npy".format(path, seed), val_losses)
    np.save("./{}/pretraining_lrs_{}.npy".format(path, seed), lrs)
    print(f"Epoch {epoch+1}: loss = {train_loss:.6f}\t val loss = {val_loss:.6f}")

    # Save the embeddings
    if(seed == 0):
        embs = []
        all_coeffs = []
        all_corr = []
        with torch.no_grad():
            #for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
            for bn, (x0, grid, sentence_embeddings) in enumerate(train_loader):
                x0 = x0[:,mask].permute(0,2,3,1)
                grid = grid.permute(0,2,3,1)

                emb, corr = transformer(x0.cuda(), grid.cuda(), sentence_embeddings, return_embedding=True)

                embs.append(emb)
                if(list(corr.shape) == [config['pretraining_batch_size'], config['pretraining_batch_size']]):
                    all_corr.append(corr.unsqueeze(0))
                
                #print(x0.shape)
                #print(x0[:,0,0,-5:].cpu().shape)
                all_coeffs.append(x0[:,0,0,-5:].cpu())

        all_embs = torch.cat(embs, dim=0)
        all_corr = torch.cat(all_corr, dim=0)
        all_coeffs = torch.cat(all_coeffs, dim=0)
        np.save("./{}/pretraining_train_embeddings_{}.npy".format(path, seed), all_embs.cpu().numpy())
        np.save("./{}/pretraining_train_corr_{}.npy".format(path, seed), all_corr.cpu().numpy())
        np.save("./{}/pretraining_train_coeffs_{}.npy".format(path, seed), all_coeffs.cpu().numpy())

        embs = []
        all_coeffs = []
        all_corr = []
        with torch.no_grad():
            #for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(val_loader):
            for bn, (x0, grid, sentence_embeddings) in enumerate(val_loader):

                x0 = x0[:,mask].permute(0,2,3,1)
                grid = grid.permute(0,2,3,1)

                emb, corr = transformer(x0.cuda(), grid.cuda(), sentence_embeddings, return_embedding=True)

                embs.append(emb)
                if(list(corr.shape) == [config['pretraining_batch_size'], config['pretraining_batch_size']]):
                    all_corr.append(corr.unsqueeze(0))
                #all_coeffs.append(x0[:,-5:].cpu().transpose(0,1))
                all_coeffs.append(x0[:,0,0,-5:].cpu())

        all_embs = torch.cat(embs, dim=0)
        all_corr = torch.cat(all_corr, dim=0)
        all_coeffs = torch.cat(all_coeffs, dim=0)
        np.save("./{}/pretraining_val_embeddings_{}.npy".format(path, seed), all_embs.cpu().numpy())
        np.save("./{}/pretraining_val_corr_{}.npy".format(path, seed), all_corr.cpu().numpy())
        np.save("./{}/pretraining_val_coeffs_{}.npy".format(path, seed), all_coeffs.cpu().numpy())

    return transformer

                
if __name__ == "__main__":

    try:
        model_name = sys.argv[1]
    except IndexError:
        print("Default model is FNO. Training FNO.")
        model_name = "fno"
    try:
        assert model_name in ['fno', 'deeponet', 'oformer', 'vit']
    except AssertionError as e:
        print("\nModel must be one of: fno, deeponet, or oformer. Model selected was: {}\n".format(model_name))
        raise

    # Load config
    with open("./configs/2d_{}_config.yaml".format(model_name), 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones

    #for pns in [10, 100, 1000, 10000]:
    #for pns in [100]:
    #for pns in [200]:
    #for pns in [1000]:
    #for pns in [9999]:
    for pns in [10000]:
        #for embedding in ['clip', 'llm']:
        #for embedding in ['oformerllm', 'oformerclip']:
        #for embedding in ['oformerllm']:
        #for embedding in ['oformerinputllm']:
        for embedding in ['oformerclip', 'oformerllm']:
            #for llm in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']:
            for llm in ['all-MiniLM-L6-v2']:
            #for llm in ['all-mpnet-base-v2']:
                #for coeff in [True, False]:
                for coeff in [True]:
                    #if(llm == 'all-mpnet-base-v2'):
                    #    config['pretraining_weight_decay'] = 1e-4
                    #    config['pretraining_learning_rate'] = 1e-6

                    train_args = config['args']
                    train_args['pretraining_num_samples'] = pns
                    train_args['embedding'] = embedding
                    train_args['llm'] = llm
                    train_args['coeff'] = coeff

                    print("\n\nPRETRAINING SAMPLES: {}\tEMBEDDING: {}\tLLM: {}\tCOEFF: {}\n\n".format(
                           train_args['pretraining_num_samples'], train_args['embedding'], train_args['llm'], train_args['coeff']))
                    #if(embedding == 'oformerllm' and llm == 'all-MiniLM-L6-v2'):
                        #print("\nSKIPPING\n")
                        #continue

                    train_args['model_name'] = model_name
                    device = train_args['device']#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    prefix = train_args['data_name'].split("_")[0] + "_" + train_args['train_style']
                    prefix += '_{}'.format(train_args['llm']) if(train_args['llm'] != None) else ''
                    prefix += '_{}'.format(train_args['embedding']) if(train_args['embedding'] != None) else ''
                    prefix += '_coeff' if(train_args['coeff']) else ''
                    if('electric' in train_args['data_name']):
                        prefix = 'electric_' + prefix
                    os.makedirs("./noise_weighted_pretrained_clip_modules/{}/{}_{}".format(
                                train_args['pretraining_num_samples'], model_name, prefix),
                                exist_ok=True)
                    shutil.copy("./configs/2d_{}_config.yaml".format(model_name),
                                "./noise_weighted_pretrained_clip_modules/{}/{}_{}/2d_{}_config.yaml".format(
                                train_args['pretraining_num_samples'], model_name, prefix, model_name))
                    shutil.copy("./pretrain_plot_progress.py",
                                "./noise_weighted_pretrained_clip_modules/{}/{}_{}/pretrain_plot_progress.py".format(
                                train_args['pretraining_num_samples'], model_name, prefix))

                    train_args = config['args']
                    train_args['pretraining_num_samples'] = pns
                    train_args['embedding'] = embedding
                    train_args['llm'] = llm
                    train_args['coeff'] = coeff
                    train_args['log_freq'] = 1

                    for seed in range(train_args['num_seeds']):
                        print("SEED: {}".format(seed))
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        train_args['seed'] = seed

                        model = run_pretraining(model_name, train_args, prefix)
                    print("Done.")

