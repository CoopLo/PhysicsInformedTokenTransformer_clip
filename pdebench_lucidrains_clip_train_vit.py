import torch
import torch.nn as nn
import yaml
import h5py
from pdebench_data_handling import FNODatasetSingle, FNODatasetMult, MultiDataset
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import shutil
from loss_funcs import LpLoss

from models.pitt import StandardPhysicsInformedTokenTransformer2D, LLMPITT2D, E2ELLMPITT2D
from models.pitt import PhysicsInformedTokenTransformer2D
from models.pitt import CLIPPhysicsInformedTokenTransformer2D

#from models.vit import VisionTransformer
from models.lucidrains_vit import ViT, CLIPViT, LLMCLIPViT

from models.oformer import OFormer2D, SpatialTemporalEncoder2D, STDecoder2D, PointWiseDecoder2D
from models.deeponet import DeepONet2D
from models.fno import FNO2d, CLIPFNO2d
from models.transolver import EmbeddingTransolver
from models.factformer import LLMFactFormer2D

from helpers import get_data, get_transformer, get_loss, get_dpot_loss, as_rollout, ar_rollout, get_pretraining_loss
from metrics import metric_func

import sys


###
# Multi-gpu 
###
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

device = 'cuda' if(torch.cuda.is_available()) else 'cpu'
DEBUG = True


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path="progress_plots", seed=None, subset=None):
    ncols = 4
    fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(5*ncols,14))
    ax[0][0].imshow(y_train_true[0,...,0].detach().cpu())
    ax[0][1].imshow(y_train_true[1,...,0].detach().cpu())
    ax[0][2].imshow(y_val_true[0,...,0].detach().cpu())
    ax[0][3].imshow(y_val_true[1,...,0].detach().cpu())

    ax[1][0].imshow(y_train_pred[0,...,0].detach().cpu())
    ax[1][1].imshow(y_train_pred[1,...,0].detach().cpu())
    ax[1][2].imshow(y_val_pred[0,...,0].detach().cpu())
    ax[1][3].imshow(y_val_pred[1,...,0].detach().cpu())

    ax[0][0].set_ylabel("VALIDATION SET TRUE")
    ax[1][0].set_ylabel("VALIDATION SET PRED")
    fname = str(ep)
    plt.tight_layout()
    while(len(fname) < 8):
        fname = '0' + fname
    if(seed is not None): 
        if(subset != 'heat,adv,burger' and subset is not None):
            plt.savefig("./{}/{}_{}_{}.png".format(path, subset, seed, fname))
        else:
            plt.savefig("./{}/{}_{}.png".format(path, seed, fname))
    else:
        plt.savefig("./{}/{}.png".format(path, fname))
    plt.close()


def val_plots(ep, val_loader, preds, path="progress_plots", seed=None):
    im_num = 0
    for vals in val_loader:
        for idx, v in tqdm(enumerate(vals[1])):

            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(v.reshape(100,).detach().cpu())
            ax.plot(preds[0][idx].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./val_2/{}_{}.png".format(seed, fname))
            plt.close()

            im_num += 1


def save_embeddings(config, path, transformer, loader, train=True, seed=0):
    embs = []
    all_coeffs = []
    all_sim_mats = []
    transformer.eval()
    print("\nSAVING {} EMBEDDINGS\n".format("TRAIN" if(train) else "TEST"))
    with torch.no_grad():
        for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(loader):

            steps = torch.Tensor(np.random.choice(range(config['initial_step'], config['sim_time']), x0.shape[0])).long()
            y = torch.cat([x0[idx,:,:,i][None] for idx, i in enumerate(steps)], dim=0)
            x0 = torch.cat([x0[idx,:,:,i-config['initial_step']:i][None] for idx, i in enumerate(steps)], dim=0).flatten(3,4)
            t = None
            if(config['time']):
                #sentence_embeddings = torch.cat([sentence_embeddings[idx,i][None] for idx, i in enumerate(steps)], dim=0)
                if(not config['sentence']):
                    sentence_embeddings = torch.cat([sentence_embeddings[idx,i][None] for idx, i in enumerate(steps)], dim=0)
                else:
                    sentence_embeddings = [sentence_embeddings[i][idx] for idx, i in enumerate(steps)]


            # Put data on correct device
            if(config['coeff']): # Stack coefficients
                # Fix size of coefficients
                original_coeffs = coeffs.clone()
                coeffs = coeffs[:,None,None,:].tile(1, x0.shape[1], x0.shape[2], 1)
        
                # Stack coefficients onto values
                if(len(x0.shape) == 5): # Hacked together
                    x0 = x0.flatten(3,4)
                x0 = torch.cat((x0, coeffs), dim=-1)#.permute(0,3,1,2)
            else:
                original_coeffs = coeffs.clone()
    
            # Forward pass
            #y_pred = transformer(x0, sentence_embeddings, clip=True, ep=ep)
            if(isinstance(transformer, LLMFactFormer2D)):
                emb, sim_mat = transformer(x0, grid, 1, sentence_embeddings, return_embedding=True)
            elif(isinstance(transformer, CLIPFNO2d)):
                emb, sim_mat = transformer(x0, grid, sentence_embeddings, return_embedding=True)
            else:
                emb, sim_mat = transformer(x0, sentence_embeddings, return_embedding=True)
            embs.append(emb)
            all_coeffs.append(coeffs)
            all_sim_mats.append(sim_mat.unsqueeze(0))

        ##data = loader.dataset.data
        ##coeff = loader.dataset.coeff
        ### Do batch per equation -> Embeddings look like trash when all from same equation
        ##for idx in range(loader.dataset.coeff.shape[0]):
        ##    
        ##    start = idx * len(loader.dataset.dsets[idx].sentence_embeddings)
        ##    end = (idx+1) * len(loader.dataset.dsets[idx].sentence_embeddings)

        ##    inp = data[start:end]
        ##    #print(inp.shape)
        ##    for i in range(inp.shape[3] - config['initial_step']):
        ##        x0 = inp[...,i:i+config['initial_step'],:].flatten(3,4)

        ##        c = coeff[idx].unsqueeze(0)
        ##        s_emb = loader.dataset.dsets[idx].sentence_embeddings

        ##        # Stack coefficients
        ##        if(config['coeff']): # Stack coefficients
        ##            # Fix size of coefficients
        ##            coeffs = coeffs[:,None,None,:].tile(1, x0.shape[1], x0.shape[2], 1)
        ##        
        ##            # Stack coefficients onto values
        ##            if(len(x0.shape) == 5): # Hacked together
        ##                x0 = x0.flatten(3,4)
        ##            x0 = torch.cat((x0, coeffs), dim=-1)#.permute(0,3,1,2)
        ##        else:
        ##            if(len(x0.shape) == 5): # Hacked together
        ##                x0 = x0.flatten(3,4)

        ##        emb, sim_mat = transformer(x0, s_emb, return_embedding=True)
        ##        embs.append(emb)
        ##        all_coeffs.append(c)
        ##        all_sim_mats.append(sim_mat.unsqueeze(0))

    split = "train" if(train) else "val"

    all_embs = torch.cat(embs, dim=0)
    np.save("./{}/pretraining_{}_embeddings_{}.npy".format(path, split, seed), all_embs.cpu().numpy())

    all_coeffs = torch.cat(all_coeffs, dim=0)
    np.save("./{}/pretraining_{}_coeffs_{}.npy".format(path, split, seed), all_coeffs.cpu().numpy())

    all_sim_mats = [a for a in all_sim_mats if(a.shape == all_sim_mats[0].shape)]
    all_sim_mats = torch.cat(all_sim_mats, dim=0)
    np.save("./{}/pretraining_{}_sim_mats_{}.npy".format(path, split, seed), all_sim_mats.cpu().numpy())


#def run_pretraining(config, prefix, model="vit"):
def run_pretraining(rank, world_size, config, prefix, model="vit", seed=0):
    print("SETTING UP DDP")
    #ddp_setup(rank, world_size)

    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    pretrained_path = "{}{}/{}".format(config['pretrained_model_path'], config['pretraining_num_samples'],
                                       prefix)

    model_name = 'pretraining' + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    pretrained_model_path = pretrained_path + "/" + model_name

    # Create the transformer model.
    transformer = get_transformer(model, config)
    #model = DDP(transformer, device_ids=[0])

    if(config['pretraining_num_samples'] == 0):
        print("\nNO PRETRAINING\n")
        return transformer, None
    if(config['load_pretrained']):
        try:
            transformer.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
            print("\nSUCCESSFULLY LOADED PRETRAINED MODEL\n")
            return transformer, pretrained_model_path
        except:
            print("\nNO PRETRAINED MODEL FOUND AT: {}. RUNNING PRETRAINIG.\n".format(pretrained_model_path))

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    #train_loader, val_loader, test_loader = new_get_data(config, pretraining=True)
    train_loader, val_loader, test_loader = get_data(config, pretraining=True)

    ################################################################
    # training and evaluation
    ################################################################

    _data, _, _, _, _ = next(iter(train_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)

    # Use Adam as the optimizer.
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['pretraining_learning_rate'],
                                 weight_decay=config['pretraining_weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['pretraining_learning_rate'],# div_factor=1e6,
                                                    steps_per_epoch=len(train_loader), epochs=config['pretraining_epochs'])

    # Use mean squared error as the loss function.
    loss_fn = nn.CrossEntropyLoss(reduction='mean') if(config['pretraining_loss'] == 'clip') else nn.L1Loss(reduction='mean')

    # Train the transformer for the specified number of epochs.
    train_losses = []
    val_losses = []
    loss_val_min = np.infty
    #src_mask = generate_square_subsequent_mask(640).cuda()
    lrs = []
    shift = 0
    print("\nPRETRAINNIG...")
    for epoch in tqdm(range(config['pretraining_epochs'])):
        # Iterate over the training dataset.
        train_loss = 0
        max_val = 0
        transformer.train()

        #train_loader.sampler.set_epoch(epoch)

        for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(train_loader):
            start = time.time()
            sentence_embeddings = sentence_embeddings if(isinstance(sentence_embeddings, list)) else \
                                  sentence_embeddings.to(device='cuda:1') if(not isinstance(transformer, (CLIPFNO2d, LLMFactFormer2D, LLMPITT2D))) else \
                                  sentence_embeddings.to(device='cuda:0')
            loss = get_pretraining_loss(
                          config,
                          transformer,
                          x0.to(device='cuda:0'),
                          grid.to(device='cuda:0'),
                          coeffs.to(device='cuda:0'),
                          #sentence_embeddings.to(device='cuda:1'),
                          sentence_embeddings,
                          loss_fn,
                          times=train_loader.dataset.dt.cuda(),
                          ep=epoch
            )

            del x0
            del grid
            del coeffs
            torch.cuda.empty_cache()

            # Do optimizer and scheduler steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= (bn + 1)
        train_losses.append(train_loss)

        with torch.no_grad():
            transformer.eval()
            val_loss = 0
            for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(val_loader):
                sentence_embeddings = sentence_embeddings if(isinstance(sentence_embeddings, list)) else \
                                      sentence_embeddings.to(device='cuda:1') if(not isinstance(transformer, (CLIPFNO2d, LLMFactFormer2D, LLMPITT2D))) else \
                                      sentence_embeddings.to(device='cuda:0')
                loss = get_pretraining_loss(
                              config,
                              transformer,
                              x0.to(device='cuda:0'),
                              grid.to(device='cuda:0'),
                              coeffs.to(device='cuda:0'),
                              #sentence_embeddings.to(device='cuda:1'),
                              sentence_embeddings,
                              loss_fn,
                              times=val_loader.dataset.dt.cuda(),
                              ep=epoch
                )
                val_loss += loss.item()

                del x0
                del grid
                del coeffs
                torch.cuda.empty_cache()

            # Save best model so far
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

    os.makedirs(pretrained_path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val_min
        }, pretrained_model_path)

    # Save the embeddings
    if(seed == 0):
        #if(config['train_style'] == 'fixed_future'):
        save_embeddings(config, path, transformer, train_loader, seed=seed, train=True)
        save_embeddings(config, path, transformer, val_loader, seed=seed, train=False)

    print("\nDONE WITH SMALL SCALE TESTING\n")
    return transformer, pretrained_model_path


def evaluate(test_loader, transformer, loss_fn, config=None):
    metrics = {'RMSE': [], 'nRMSE': [], 'CSV': [], 'Max': [], 'BD': [], 'F': []}
    with torch.no_grad():
        transformer.eval()
        test_loss = 0
        for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(test_loader):
            #y_pred, y, loss, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = get_loss(config, transformer, x0, grid, coeffs,
            #                                                                           loss_fn,
            #                                                                           sentence_embeddings=sentence_embeddings,
            #                                                                           times=test_loader.dataset.dt,
            #                                                                           evaluate=True)
            if(not isinstance(transformer, E2ELLMPITT2D)):
                sentence_embeddings = sentence_embeddings if(isinstance(sentence_embeddings, list)) else \
                                  sentence_embeddings.to(device='cuda:1') if(not isinstance(transformer, (CLIPFNO2d, LLMFactFormer2D, LLMPITT2D, E2ELLMPITT2D))) else \
                                  sentence_embeddings.to(device='cuda:0')
                                  #sentence_embeddings.to(device='cuda:1') if(not isinstance(transformer, LLMPITT2D)) else \
            y_pred, y, loss, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = get_dpot_loss(config, 1, transformer,
                                                                                    x0.to(device='cuda:0'),
                                                                                    grid.to(device='cuda:0'),
                                                                                    coeffs.to(device='cuda:0'),
                                                                                    loss_fn,
                                                                                    sentence_embeddings=sentence_embeddings,
                                                                                    times=test_loader.dataset.dt.to(device='cuda:0'),
                                                                                    evaluate=True)
            test_loss += loss.item()

            metrics['RMSE'].append(err_RMSE)
            metrics['nRMSE'].append(err_nRMSE)
            metrics['CSV'].append(err_CSV)
            metrics['Max'].append(err_Max)
            metrics['BD'].append(err_BD)
            metrics['F'].append(err_F)

    return test_loss/(bn+1), metrics


def zero_shot_evaluate(transformer, config, seed, prefix, subset='Heat,Burger,Adv'):
    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    train_loader, val_loader, test_loader = get_data(config)
    metrics = {'RMSE': [], 'nRMSE': [], 'CSV': [], 'Max': [], 'BD': [], 'F': []}
    loss_fn = LpLoss(2,2)
    print("\nEVALUATING...")
    with torch.no_grad():
        transformer.eval()
        test_loss = 0
        for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(test_loader):
            #y_pred, y, loss, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = get_loss(config, transformer, x0, grid, coeffs,
            #                                                                           loss_fn,
            #                                                                           sentence_embeddings=sentence_embeddings,
            #                                                                           times=test_loader.dataset.dt,
            #                                                                           evaluate=True)
            y_pred, y, loss, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = get_dpot_loss(config, 1, transformer, x0, grid,
                                                                                       coeffs,
                                                                                       loss_fn,
                                                                                       sentence_embeddings=sentence_embeddings,
                                                                                       times=test_loader.dataset.dt,
                                                                                       evaluate=True)
            test_loss += loss.item()

            metrics['RMSE'].append(err_RMSE)
            metrics['nRMSE'].append(err_nRMSE)
            metrics['CSV'].append(err_CSV)
            metrics['Max'].append(err_Max)
            metrics['BD'].append(err_BD)
            metrics['F'].append(err_F)

    if(subset != 'heat,adv,burger'):
        np.save("./{}/zero_shot/zero_shot_{}_test_vals_{}.npy".format(path, subset, seed), test_loss/(bn+1))
        np.save("./{}/zero_shot/zero_shot_{}_metrics_{}.npy".format(path, subset, seed), metrics)
    else:
        np.save("./{}/zero_shot/zero_schot_test_vals_{}.npy".format(path, seed), test_loss/(bn+1))
        np.save("./{}/zero_shot/zero_schot_metrics_{}.npy".format(path, seed), metrics)
    return test_loss/(bn+1)


def freeze_llm(transformer, train_loader, val_loader, test_loader):
    if(isinstance(train_loader.dataset, MultiDataset)):
        print("Updating train loader...")
        for dset in tqdm(train_loader.dataset.dsets):
            s_emb = []
            for sentence in dset.sentence_embeddings:
                s_emb.append(transformer._llm_forward([sentence]).detach())
            s_emb = torch.cat(s_emb, dim=0)
            dset.sentence_embeddings = s_emb

        print("Updating val loader...")
        for dset in tqdm(val_loader.dataset.dsets):
            s_emb = []
            for sentence in dset.sentence_embeddings:
                s_emb.append(transformer._llm_forward([sentence]).detach())
            s_emb = torch.cat(s_emb, dim=0)
            dset.sentence_embeddings = s_emb

        print("Updating test loader...")
        for dset in tqdm(test_loader.dataset.dsets):
            s_emb = []
            for sentence in dset.sentence_embeddings:
                s_emb.append(transformer._llm_forward([sentence]).detach())
            s_emb = torch.cat(s_emb, dim=0)
            dset.sentence_embeddings = s_emb

    else:
        print("Updating train loader...")
        s_emb = []
        for sentence in train_loader.dataset.sentence_embeddings:
            s_emb.append(transformer._llm_forward([sentence]).detach())
        s_emb = torch.cat(s_emb, dim=0)
        train_loader.dataset.sentence_embeddings = s_emb

        print("Updating val loader...")
        s_emb = []
        for sentence in val_loader.dataset.sentence_embeddings:
            s_emb.append(transformer._llm_forward([sentence]).detach())
        s_emb = torch.cat(s_emb, dim=0)
        val_loader.dataset.sentence_embeddings = s_emb

        print("Updating val loader...")
        s_emb = []
        for sentence in test_loader.dataset.sentence_embeddings:
            s_emb.append(transformer._llm_forward([sentence]).detach())
        s_emb = torch.cat(s_emb, dim=0)
        test_loader.dataset.sentence_embeddings = s_emb

    return train_loader, val_loader, test_loader


def run_training(transformer, config, prefix, seed, subset='heat,adv,burger'):
    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    model_name = 'vit' + "_{}.pt".format(seed)
    if(subset != 'heat,adv,burger'):
        model_name = subset + "_" + model_name
    model_path = path + "/" + model_name

    #if(subset == 'all'):
    #    return model_path

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    train_loader, val_loader, test_loader = get_data(config)

    # Freeze LLM by updating sentence embeddings
    if(isinstance(transformer, E2ELLMPITT2D) and subset != 'all'):
        train_loader, val_loader, test_loader = freeze_llm(transformer, train_loader,
                                                           val_loader, test_loader)

    ################################################################
    # training and evaluation
    ################################################################

    _data, _, _, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)

    # Use Adam as the optimizer.
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
                                                    steps_per_epoch=len(train_loader), epochs=config['epochs'])

    # Use mean squared error as the loss function.
    loss_fn = LpLoss(2,2)

    # Train the transformer for the specified number of epochs.
    train_losses = []
    val_losses = []
    loss_val_min = np.infty
    lrs = []
    shift = 0
    print("\nTRAINING...")
    for epoch in tqdm(range(config['epochs'])):
        # Iterate over the training dataset.
        train_loss = 0
        max_val = 0
        transformer.train()
        for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(train_loader):
            start = time.time()
            #y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, loss_fn,
            #                           sentence_embeddings=sentence_embeddings,
            #                           times=train_loader.dataset.dt)
            if(isinstance(sentence_embeddings, torch.Tensor)):
                sentence_embeddings = sentence_embeddings if(isinstance(sentence_embeddings, list)) else \
                                      sentence_embeddings.to(device='cuda:1') if(not isinstance(transformer, (CLIPFNO2d, LLMFactFormer2D, LLMPITT2D, E2ELLMPITT2D))) else \
                                      sentence_embeddings.to(device='cuda:0')
            y_pred, y, loss = get_dpot_loss(
                                         config,
                                         epoch,
                                         transformer,
                                         x0.to(device='cuda:0'),
                                         grid.to(device='cuda:0'),
                                         coeffs.to(device='cuda:0'),
                                         loss_fn,
                                         sentence_embeddings=sentence_embeddings,
                                         times=train_loader.dataset.dt.to(device='cuda:0'))

            # Backward pass: compute gradient of the loss with respect to model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])

            train_loss += loss.item()
            if(bn == 0):
                y_train_true = y.clone()
                y_train_pred = y_pred.clone()

            scheduler.step()

        train_loss /= (bn + 1)
        train_losses.append(train_loss)

        if((epoch%config['validate'] == 0) or ((epoch+1) == config['epochs'])):
            #print("VALIDATING")
            with torch.no_grad():
                transformer.eval()
                val_loss = 0
                all_val_preds = []
                for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(val_loader):
                    #y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, loss_fn,
                    #                           sentence_embeddings=sentence_embeddings,
                    #                           times=train_loader.dataset.dt)
                    #y_pred, y, loss = get_dpot_loss(config, epoch, transformer, x0, grid, coeffs, loss_fn,
                    #                           sentence_embeddings=sentence_embeddings,
                    #                           times=train_loader.dataset.dt)
                    if(isinstance(sentence_embeddings, torch.Tensor)):
                        sentence_embeddings = sentence_embeddings if(isinstance(sentence_embeddings, list)) else \
                                              sentence_embeddings.to(device='cuda:1') if(not isinstance(transformer, (CLIPFNO2d, LLMFactFormer2D, LLMPITT2D, E2ELLMPITT2D))) else \
                                              sentence_embeddings.to(device='cuda:0')
                    y_pred, y, loss = get_dpot_loss(
                                                 config,
                                                 epoch,
                                                 transformer,
                                                 x0.to(device='cuda:0'),
                                                 grid.to(device='cuda:0'),
                                                 coeffs.to(device='cuda:0'),
                                                 loss_fn,
                                                 sentence_embeddings=sentence_embeddings,
                                                 times=train_loader.dataset.dt.to(device='cuda:0'))

                    all_val_preds.append(y_pred.detach())

                    val_loss += loss.item()
                    if(bn == 0):
                        y_val_true = y.clone()
                        y_val_pred = y_pred.clone()

                val_loss /= (bn + 1)
                if  val_loss < loss_val_min:
                    loss_val_min = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': transformer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_val_min
                        }, model_path)

            val_losses.append(val_loss)

        # Print the loss at the end of each epoch.
        if(epoch%config['log_freq'] == 0):
            np.save("./{}/{}_train_l2s_{}.npy".format(path, subset, seed), train_losses)
            np.save("./{}/{}_val_l2s_{}.npy".format(path, subset, seed), val_losses)
            np.save("./{}/{}_lrs_{}.npy".format(path, subset, seed), lrs)
            print(f"Epoch {epoch+1}: loss = {train_loss:.6f}\t val loss = {val_loss:.6f}")

        if(epoch%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed, subset=subset)

    progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed, subset=subset)

    test_vals = []
    eval_loss_fn = LpLoss(2,2)
    test_value, last_metric = evaluate(test_loader, transformer, eval_loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))

    transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value, best_metric = evaluate(test_loader, transformer, eval_loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    #if(subset != 'heat,adv,burger'):
    np.save("./{}/{}_test_vals_{}.npy".format(path, subset, seed), test_vals)
    np.save("./{}/metrics/{}_last_metrics_{}.npy".format(path, subset, seed), last_metric)
    np.save("./{}/metrics/{}_best_metrics_{}.npy".format(path, subset, seed), best_metric)
    #else:
    #    np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)
    #    np.save("./{}/metrics/last_metric_{}.npy".format(path, seed), last_metric)
    #    np.save("./{}/metrics/best_metric_{}.npy".format(path, seed), best_metric)

    if(config['train_style'] == 'arbitrary_step'):
        as_rollout(test_loader, transformer, loss_fn, config, prefix, subset, seed=seed)
    elif(config['train_style'] == 'next_step'):
        print("\nPREFIX: {}\n".format(prefix))
        #ar_rollout(test_loader, transformer, loss_fn, config, prefix, subset, seed=seed)

    return model_path


if __name__ == '__main__':
    # Create a transformer with an input dimension of 10, a hidden dimension
    # of 20, 2 transformer layers, and 8 attention heads.

    # Load config
    #raise
    if(len(sys.argv) == 1):
        raise ValueError("Select one of vit, transovler, or pitt.")
    elif(sys.argv[1] == 'transolver'):
        model_name = 'transolver'
        config_name = "transolver_2d_config.yaml"
    elif(sys.argv[1] == 'vit'):
        #model_name = 'clipvit'
        model_name = 'llmvit'
        config_name = "lucidrains_2d_vit_config.yaml"
    elif(sys.argv[1] == 'pitt'):
        model_name = 'pitt'
        config_name = "pitt_2d_config.yaml"
    elif(sys.argv[1] == 'dpot'):
        model_name = 'llmdpot'
        config_name = "dpot_2d_config.yaml"
    elif(sys.argv[1] == 'factformer'):
        model_name = 'llmfactformer'
        config_name = "factformer_2d_config.yaml"
    elif(sys.argv[1] == 'fno'):
        model_name = 'llmfno'
        config_name = "fno_2d_config.yaml"
    else:
        print("Using ViT by default.")
        model_name = 'vit'
        config_path = "lucidrains_2d_vit_config.yaml"

    with open("./configs/{}".format(config_name), 'r') as stream:
        config = yaml.safe_load(stream)

    # Pick between full LLM training and just projection head
    if(sys.argv[1] == 'vit' and config['args']['sentence']):
        model_name = 'llmvit'
    elif(sys.argv[1] == 'vit' and not config['args']['sentence']):
        model_name = 'clipvit'

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    prefix = "2D_{}_".format(model_name) + train_args['train_style'] + "_" + train_args['dataset'] + "_" + \
             train_args['pretraining_loss'] + "_" + train_args['llm']
    prefix += "_bcs" if(train_args['bcs']) else ""
    prefix += "_coeff" if(train_args['coeff']) else ""
    prefix += "_eqcoeff" if(train_args['eq_coeff']) else ""
    prefix += "_transfer" if(train_args['transfer']) else ""
    prefix += "_sentence" if(train_args['sentence']) else ""
    prefix += "_qualitative" if(train_args['qualitative']) else ""
    prefix += "_time" if(train_args['time']) else ""
    prefix += "_DEBUG" if(train_args['DEBUG']) else ""

    train_args['prefix'] = prefix

    if(train_args['dataset'] == 'all'):
        train_args['sim_time'] = 21

    # Loop over number of samples TODO: ns = -1 is not supported in autoregressive rollout
    #for ns in [10, 20, 50, 100]:
    #for ns in [10]:
    #for ns in [10, 20, 50, 100, 200]:#, 100]:
    #for ns in [100, 200]:
    #for ns in [10, 20, 50, 100, 200, 500]:
    #for ns in [20, 50]:#, 100]:
    #for ns in [50]:#, 100]:
    #for ns in [10]:
    #for ns in [50]:
    for ns in [100]:
    #for ns in [1000]:

        train_args['num_samples'] = ns
        train_args['num_data_samples'] = ns

        # Creat save directory
        os.makedirs("{}{}_{}/{}".format(train_args['results_dir'], train_args['num_samples'],
                                            train_args['pretraining_num_samples'], prefix), exist_ok=True)
        os.makedirs("{}{}_{}/{}/metrics".format(train_args['results_dir'], train_args['num_samples'],
                                            train_args['pretraining_num_samples'], prefix), exist_ok=True)
        os.makedirs("{}{}_{}/{}/zero_shot".format(train_args['results_dir'], train_args['num_samples'],
                                            train_args['pretraining_num_samples'], prefix), exist_ok=True)
        os.makedirs("{}{}_{}/{}/rollouts".format(train_args['results_dir'], train_args['num_samples'],
                                            train_args['pretraining_num_samples'], prefix), exist_ok=True)

        # Copy files to save directory
        shutil.copy("./configs/{}".format(config_name),
                    "{}{}_{}/{}/{}".format(train_args['results_dir'], train_args['num_samples'],
                                           train_args['pretraining_num_samples'], prefix, config_name))
        shutil.copy("./plot_progress.py", "{}{}_{}/{}/plot_progress.py".format(train_args['results_dir'], train_args['num_samples'],
                                                                                   train_args['pretraining_num_samples'], prefix))
        shutil.copy("./pretrain_plot_progress.py", "{}{}_{}/{}/pretrain_plot_progress.py".format(train_args['results_dir'],
                                 train_args['num_samples'], train_args['pretraining_num_samples'], prefix))
        shutil.copy("./finetune_plot_progress.py", "{}{}_{}/{}/finetune_plot_progress.py".format(train_args['results_dir'],
                                 train_args['num_samples'], train_args['pretraining_num_samples'], prefix))

        for seed in range(train_args['num_seeds']):
            train_args['num_samples'] = ns
            print("\nSEED: {}\n".format(seed))
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_args['seed'] = seed
            train_args['num_samples'] = ns
            train_args['num_data_samples'] = ns

            # Will run pretraining if num_pretraining_samples > 0
            print("\n\nMODEL NAME: {}\n\n".format(model_name))

            if(train_args['DEBUG']):
                pretrained_model_path = None
            else:
                model, pretrained_model_path = run_pretraining(1, 1, train_args, prefix, model=model_name)

                #world_size = torch.cuda.device_count()
                #print("WORLD SIZE: {}".format(world_size))
                #mp.spawn(run_pretraining, args=(world_size, train_args, prefix, model_name, seed,), nprocs=world_size)

                model.finished_pretraining()
                print("\n\nPRETRAINED MODEL PATH: {}\n\n".format(pretrained_model_path))

            torch.manual_seed(seed)
            np.random.seed(seed)
            model = get_transformer(model_name, train_args)
            model.finished_pretraining()
            if(pretrained_model_path is not None):
                model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])

            # Train on combined dataset
            train_args['dataset'] = 'all'
            if(not train_args['DEBUG']):
                transfer_model_path = run_training(model, train_args, prefix, seed, subset=train_args['dataset'])

            #if(train_args['transfer']):
            #for subset in ['diffusion_reaction', 'cfd_rand_0.1_0.01_0.01', 'cfd_rand_0.1_0.1_0.1',
            #for subset in ['cfd_rand_0.1_0.01_0.01', 'cfd_rand_0.1_0.1_0.1',
            #for subset in ['shallow_water']:
            for subset in ['shallow_water', 'diffusion_reaction', 'cfd_rand_0.1_0.01_0.01', 'cfd_rand_0.1_0.1_0.1',
                           'cfd_rand_0.1_1e-8_1e-8', 'cfd_rand_1.0_0.01_0.01', 'cfd_rand_1.0_0.1_0.1', 'cfd_rand_1.0_1e-8_1e-8',
                           'cfd_turb_0.1_1e-8_1e-8', 'cfd_turb_1.0_1e-8_1e-8', 'heat', 'burger', 'adv']:

                torch.cuda.empty_cache()
                if(train_args['DEBUG']):
                    train_args['dataset'] = 'cfd_rand_0.1_0.01_0.01'
                    train_args['num_data_samples'] = 50
                else:
                    print("\nDATA: {}\n".format(subset))
                    train_args['dataset'] = subset
                    train_args['num_data_samples'] = 1000
                    #train_args['num_data_samples'] = 500
                    #train_args['num_data_samples'] = 50

                ###
                #  Either load from transfer learning path (standard training on combined data set) or from pretrained path
                #  TODO: Maybe explore pretraining THEN transfer learning?
                ###
                if(train_args['transfer']):
                    print("\nTRANSFER LEARNING FROM: {}\n".format(transfer_model_path))
                    model.load_state_dict(torch.load(transfer_model_path)['model_state_dict'])
                elif(pretrained_model_path is not None):
                    print("\nFINE TUNING FROM: {}\n".format(pretrained_model_path))
                    model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
                torch.cuda.empty_cache()

                #if(not train_args['DEBUG']):
                #    print("\nDOING ZERO-SHOT EVALUATION\n")
                #    zero_shot_evaluate(model, train_args, seed, prefix, subset=train_args['dataset'])
                torch.cuda.empty_cache()

                print("\nFINE TUNING ON INDIVIDUAL DATA SET\n")
                model_path = run_training(model, train_args, prefix, seed, subset=train_args['dataset'])
                torch.cuda.empty_cache()

                if(train_args['DEBUG']):
                    break

