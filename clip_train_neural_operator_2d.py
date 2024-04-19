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
from models.oformer import SpatialTemporalEncoder2D, PointWiseDecoder2D, OFormer2D, STDecoder2D
from models.fno import CLIPFNO2d, FNO2d
from models.deeponet import DeepONet2D
from models.vit import VisionTransformer

from utils import TransformerOperatorDataset2D, ElectricTransformerOperatorDataset2D
from anthony_data_handling import PDEDataset2D

import yaml
from tqdm import tqdm
import h5py
from matplotlib import pyplot as plt
import random

def progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path="progress_plots", seed=None):
    ncols = 4
    fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(5*ncols,14))
    ax[0][0].imshow(y_train_true[0].detach().cpu())
    ax[0][1].imshow(y_train_true[1].detach().cpu())
    ax[0][2].imshow(y_val_true[0].detach().cpu())
    ax[0][3].imshow(y_val_true[1].detach().cpu())

    ax[1][0].imshow(y_train_pred[0].detach().cpu())
    ax[1][1].imshow(y_train_pred[1].detach().cpu())
    ax[1][2].imshow(y_val_pred[0].detach().cpu())
    ax[1][3].imshow(y_val_pred[1].detach().cpu())

    ax[0][0].set_xlabel("VALIDATION SET TRUE")
    ax[1][0].set_xlabel("VALIDATION SET PRED")
    #ax[0][2].set_title("VALIDATION SET PRED")
    #ax[0][3].set_title("VALIDATION SET PRED")
    fname = str(ep)
    while(len(fname) < 8):
        fname = '0' + fname
    if(seed is not None):
        plt.savefig("./{}/{}_{}.png".format(path, seed, fname))
    else:
        plt.savefig("./{}/{}.png".format(path, fname))
    plt.close()


def val_plots(ep, val_loader, preds, path="progress_plots", seed=None):
    im_num = 0
    for vals in val_loader:
        for idx, v in tqdm(enumerate(vals[1])):

            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(v.reshape(200,).detach().cpu())
            #print(preds[0].shape)
            #ax.plot(preds[0][idx,:,0,0].detach().cpu())
            ax.plot(preds[0][idx].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./val_1/{}_{}.png".format(seed, fname))
            plt.close()

            im_num += 1


def train_plots(train_loader, split, seed=None):
    im_num = 0
    viscocities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    amplitudes = [0.01, 0.05, 0.1, 0.25, 0.5, 1.]
    for vals in train_loader:
        #print(vals[2][0][1:] - vals[2][0][:-1])
        #raise
        #for idx, v in tqdm(enumerate(vals[1])):
        for idx, v in enumerate(vals[1]):
            visc = viscocities[(im_num//100)//6]
            amp = amplitudes[(im_num//100)%6]
            print(im_num, (im_num//100), (im_num//100)//7, (im_num//100)%6)

            fig, ax = plt.subplots(figsize=(8,6))
            ax.imshow(v.detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            ax.set_title("{} Viscocity, {} Amplitude".format(visc, amp))
            plt.savefig("./{}_plots/{}_{}.png".format(split, seed, fname))
            plt.close()
            #raise

            im_num += 1


def get_model(model_name, config):
    if(model_name == "fno"):
        model = CLIPFNO2d(
                    config['num_channels'],
                    config['modes1'],
                    config['modes2'],
                    config['width'],
                    config['initial_step']+5 if(config['coeff']) else config['initial_step'],
                    config['dropout'],
                    embed_dim=config['embed_dim'],
        )
    elif(model_name == "oformer"):
        encoder = SpatialTemporalEncoder2D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], heads=config['heads'])
                            #, dropout=config['dropout'],
                            #res=config['enc_res'])
        decoder = PointWiseDecoder2D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     propagator_depth=config['decoding_depth'], scale=config['scale'], out_steps=1)
        #decoder = STDecoder2D(latent_channels=config['latent_channels'], out_channels=config['out_channels'], out_steps=1,
        #                       propagator_depth=config['decoder_depth'], scale=config['scale'], res=config['dec_res'])
        model = OFormer2D(encoder, decoder, num_x=config['num_x'], num_y=config['num_y'])
    elif(model_name == "deeponet"):
        model = DeepONet2D(config['branch_net'], config['trunk_net'], config['activation'], config['kernel_initializer'])
    elif(model_name == 'vit'):
        model = VisionTransformer(
                   img_size=config['img_size'],
                   patch_size=config['patch_size'],
                   #in_chans=config['initial_step']+2,
                   in_chans=config['initial_step']+7 if(config['coeff']) else config['initial_step']+2,
                   out_chans=1,
                   embed_dim=config['embed_dim'],
                   depth=config['depth'],
                   n_heads=config['n_heads'],
                   mlp_ratio=config['mlp_ratio'],
                   qkv_bias=config['qkv_bias'],
                   drop_rate=config['drop_rate'],
                   attn_drop_rate=config['attn_drop_rate'],
                   stride=config['patch_stride'],
        )
    
    model.to(device)
    return model


def get_data(f, config):
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
    print("\nTRAINING DATA")
    if('electric' in config['data_name']):
        train_data = ElectricTransformerOperatorDataset2D(f,
                                split="train",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                split_style=config['split_style'],
                                seed=config['seed'],
        )
        print("\nVALIDATION DATA")
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        val_data = ElectricTransformerOperatorDataset2D(f,
                                split="val",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                split_style=config['split_style'],
                                seed=config['seed'],
        )
        print("\nTEST DATA")
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        test_data = ElectricTransformerOperatorDataset2D(f,
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                split_style=config['split_style'],
                                seed=config['seed'],
        )
    else:
        train_data = TransformerOperatorDataset2D(f,
                                split="train",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                split_style=config['split_style'],
                                samples_per_equation=config['samples_per_equation'],
                                seed=config['seed'],
        )
        print("\nVALIDATION DATA")
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        val_data = TransformerOperatorDataset2D(f,
                                split="val",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                split_style=config['split_style'],
                                samples_per_equation=config['samples_per_equation'],
                                seed=config['seed'],
        )
        print("\nTEST DATA")
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        test_data = TransformerOperatorDataset2D(f,
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                split_style=config['split_style'],
                                samples_per_equation=config['samples_per_equation'],
                                seed=config['seed'],
        )

    if(config['split_style'] == 'equation'):
        assert not (bool(set(train_data.data_list) & \
                         set(val_data.data_list)) | \
                    bool(set(train_data.data_list) & \
                         set(test_data.data_list)) & \
                    bool(set(val_data.data_list) & \
                         set(test_data.data_list)))
    elif(config['split_style'] == 'initial_condition'):
        assert not (bool(set(train_data.idxs) & \
                         set(val_data.idxs)) | \
                    bool(set(train_data.idxs) & \
                         set(test_data.idxs)) & \
                    bool(set(val_data.idxs) & \
                         set(test_data.idxs)))
    else:
        raise ValueError("Invalid splitting style. Select initial_condition or equation")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                               num_workers=config['num_workers'], shuffle=True,
                                               generator=torch.Generator(device='cuda'))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=True,
                                             generator=torch.Generator(device='cuda'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False,
                                             generator=torch.Generator(device='cuda'))
    return train_loader, val_loader, test_loader


def new_get_data(f, config, pretraining=False):
    train_data = PDEDataset2D(
            path="/home/cooperlorsung/2d_heat_adv_burgers_train_large.h5",
            pde="Heat, Burgers, Advection",
            mode="train",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=False,
            device='cuda:0',
            num_samples=config['pretraining_num_samples'] if(pretraining) else config['num_samples'],
            clip=config['embedding'] == 'clip',
            downsample=config['downsample'],
    )
    val_data = PDEDataset2D(
            path="/home/cooperlorsung/2d_heat_adv_burgers_valid_large.h5",
            pde="Heat, Burgers, Advection",
            mode="valid",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=True,
            device='cuda:0',
            num_samples=config['num_samples'],
            clip=config['embedding'] == 'clip',
            downsample=config['downsample'],
    )
    test_data = PDEDataset2D(
            path="/home/cooperlorsung/2d_heat_adv_burgers_test_large.h5",
            pde="Heat, Burgers, Advection",
            mode="test",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=True,
            device='cuda:0',
            num_samples=config['num_samples'],
            clip=config['embedding'] == 'clip',
            downsample=config['downsample'],
    )
    batch_size = config['pretraining_batch_size'] if(pretraining) else config['batch_size']
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, generator=torch.Generator(device='cuda'),
                                               num_workers=config['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, generator=torch.Generator(device='cuda'),
                                             num_workers=config['num_workers'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             num_workers=config['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(test_loader, model, loss_fn, navier_stokes=True, config=None):
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        model.eval()
        #for bn, (xx, yy, grid, tokens, t) in enumerate(test_loader):
        for bn, (xx, grid, coeffs, sentence_embeddings) in enumerate(test_loader):
            if(config['train_style'] == 'next_step'):
                random_step = random.randint(config['initial_step'], config['sim_time'])
                yy = xx[:, random_step].unsqueeze(-1).to(device).float()
                xx = xx[:, random_step-config['initial_step']:random_step].permute(0,2,3,1).to(device).float()
            else:
                yy = xx[:, config['sim_time']].unsqueeze(-1).to(device).float()
                xx = xx[:, :config['initial_step']].permute(0, 2, 3, 1).to(device).float()
            #xx = xx.to(device).float()
            #yy = yy.to(device).float()
            grid = grid.to(device).float()
            
            if(isinstance(model, (FNO2d, OFormer2D))):
                if(navier_stokes):
                    x = torch.swapaxes(xx, 1, 3)
                    x = torch.swapaxes(x, 1, 2)
                else:
                    x = xx
                im, loss = model.get_loss(x, yy[...,0], grid, loss_fn)
            elif(isinstance(model, DeepONet2D)):
                #if(navier_stokes):
                #    x = torch.swapaxes(xx, 1, 3)
                #    x = torch.swapaxes(x, 1, 2)
                #else:
                #    x = xx
                im = model(xx, grid)
                loss = loss_fn(yy[...,0], im)
            elif(isinstance(model, CLIPFNO2d)):
                if(config['coeff']):
                    nu = coeffs['nu'].unsqueeze(-1)
                    ax = coeffs['ax'].unsqueeze(-1)
                    ay = coeffs['ay'].unsqueeze(-1)
                    cx = coeffs['cx'].unsqueeze(-1)
                    cy = coeffs['cy'].unsqueeze(-1)
                    coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(xx.shape[0], xx.shape[1], xx.shape[2], 5)
                    inp = torch.cat((xx, coeff), dim=-1)
                else:
                    inp = xx
                im, loss = model.get_loss(inp, yy[...,0], grid, sentence_embeddings, loss_fn)
            elif(isinstance(model, VisionTransformer)):
                if(config['coeff']):
                    nu = coeffs['nu'].unsqueeze(-1)
                    ax = coeffs['ax'].unsqueeze(-1)
                    ay = coeffs['ay'].unsqueeze(-1)
                    cx = coeffs['cx'].unsqueeze(-1)
                    cy = coeffs['cy'].unsqueeze(-1)
                    coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(xx.shape[0], xx.shape[1], xx.shape[2], 5)
                    inp = torch.cat((xx, grid, coeff), dim=-1).permute(0,3,1,2)
                else:
                    inp = torch.cat((xx, grid), dim=-1).permute(0,3,1,2)

                #im = model(torch.cat((xx, grid), dim=-1).permute(0,3,1,2))
                im = model(inp)
                loss = loss_fn(yy, im)
    
            test_l2_step += loss.item()
            test_l2_full += loss.item()
    return test_l2_full/(bn+1)


def run_pretraining(model_name, config, prefix):
    #prefix = config['data_name'].split("_")[0]
    path = "{}{}_{}/{}_{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], model_name, prefix)
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
    model_path = path + "/" + model_name

    # Create the transformer model.
    print(model_name)
    transformer = get_model(model_name, config)
    if(config['pretraining_num_samples'] == 0):
        print("\nNO PRETRAINING\n")
        return transformer

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    train_loader, val_loader, test_loader = new_get_data(f, config, pretraining=True)

    ################################################################
    # training and evaluation
    ################################################################

    if(config['return_text']):
        #_data, _, _, _ = next(iter(val_loader))
        _data, _, _, _ = next(iter(val_loader))
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

    #TODO Make this step lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['pretraining_learning_rate'],# div_factor=1e6,
                                                    steps_per_epoch=len(train_loader), epochs=config['pretraining_epochs'])

    # Use mean squared error as the loss function.
    loss_fn = nn.L1Loss(reduction='mean')
    clip_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    # Train the transformer for the specified number of epochs.
    train_losses = []
    val_losses = []
    loss_val_min = np.infty
    #src_mask = generate_square_subsequent_mask(640).cuda()
    lrs = []
    shift = 0
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
        #for bn, (x0, y, grid, tokens, t, sentence_embeddings) in enumerate(train_loader):
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
            start = time.time()
            #print()
            #print(x0.shape)
            #print()
            if(config['train_style'] == 'next_step'):
                raise
            else:
                y = x0[:, config['sim_time']].unsqueeze(-1)
                x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)

            # Put data on correct device
            x0 = x0.to(device).float()
            y = y.to(device).float()

            #tokens = tokens.to(device).float()
            #t = t.to(device).float()
            grid = grid.to(device).float()
            sentence_embeddings = sentence_embeddings.to(device).float()

            # Forward pass
            if(config['coeff']):
                nu = coeffs['nu'].unsqueeze(-1)
                ax = coeffs['ax'].unsqueeze(-1)
                ay = coeffs['ay'].unsqueeze(-1)
                cx = coeffs['cx'].unsqueeze(-1)
                cy = coeffs['cy'].unsqueeze(-1)
                coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],     x0.shape[2], 5)
                inp = torch.cat((x0, coeff), dim=-1)
            else:
                inp = x0
            #y_pred = transformer(torch.cat((x0, grid), dim=-1).permute(0,3,1,2), sentence_embeddings, True)
            y_pred = transformer(inp, grid, sentence_embeddings, True)

            labels = torch.arange(y_pred.shape[1]).to(device).repeat(y_pred.shape[0], 1)
            loss = clip_loss_fn(y_pred, labels)
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
            for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(val_loader):
                # Forward pass: compute predictions by passing the input sequence
                # through the transformer.
                # Put data on correct device
                if(config['train_style'] == 'next_step'):
                    raise
                else:
                    y = x0[:, config['sim_time']].unsqueeze(-1)
                    x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)

                x0 = x0.to(device).float()
                y = y.to(device).float()
                #tokens = tokens.to(device).float()
                #t = t.to(device).float()
                grid = grid.to(device).float()

                if(config['coeff']):
                    nu = coeffs['nu'].unsqueeze(-1)
                    ax = coeffs['ax'].unsqueeze(-1)
                    ay = coeffs['ay'].unsqueeze(-1)
                    cx = coeffs['cx'].unsqueeze(-1)
                    cy = coeffs['cy'].unsqueeze(-1)
                    coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],     x0.shape[2], 5)
                    inp = torch.cat((x0, coeff), dim=-1)
                else:
                    inp = x0
                #y_pred = transformer(torch.cat((x0, grid), dim=-1).permute(0,3,1,2), sentence_embeddings, True)
                y_pred = transformer(inp, grid, sentence_embeddings, True)

                labels = torch.arange(y_pred.shape[1]).to(device).repeat(y_pred.shape[0], 1)
                loss = clip_loss_fn(y_pred, labels)
                val_loss += loss.item()

                #y = y[...,0].to(device=device)#.cuda()
                #if(bn == 0):
                #    y_val_true = y.clone()
                #    y_val_pred = y_pred.clone()
                #all_val_preds.append(y_pred.detach())

                ## Compute the loss.
                #val_loss += loss_fn(y_pred, y).item()

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

    test_vals = []
    test_value = evaluate(test_loader, transformer, loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    if(not break_pretrain):
        transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, transformer, loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("{}{}_{}/{}_{}/pretraining_test_vals_{}.npy".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], model_name, prefix, seed), test_vals)

    # Save the embeddings
    if(seed == 0):
        embs = []
        all_coeffs = []
        with torch.no_grad():
            for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
                x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)

                #emb = transformer.diff_x_proj(transformer.temporal_neural_operator(x0, grid).flatten(1,2))
                #temporal_x = x0[...,1:] - x0[...,:-1]
                if(config['coeff']):
                    nu = coeffs['nu'].unsqueeze(-1)
                    ax = coeffs['ax'].unsqueeze(-1)
                    ay = coeffs['ay'].unsqueeze(-1)
                    cx = coeffs['cx'].unsqueeze(-1)
                    cy = coeffs['cy'].unsqueeze(-1)
                    coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],     x0.shape[2], 5)
                    inp = torch.cat((x0, coeff), dim=-1)
                else:
                    inp = x0
                emb = transformer(inp, grid, sentence_embeddings, True, True)

                embs.append(emb)
                all_coeffs.append(torch.vstack(list(coeffs.values())).transpose(0,1))

        all_embs = torch.cat(embs, dim=0)
        all_coeffs = torch.cat(all_coeffs, dim=0)
        np.save("./{}/pretraining_embeddings_{}.npy".format(path, seed), all_embs.cpu().numpy())
        np.save("./{}/pretraining_coeffs_{}.npy".format(path, seed), all_coeffs.cpu().numpy())

    return transformer

                
def run_training(model, config, prefix):
    
    ################################################################
    # load data
    ################################################################
    
    #prefix = config['data_name'].split("_")[0]
    path = "{}{}_{}/{}_{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], config['model_name'], prefix)
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
    model_name = '{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    navier_stokes = not('electric' in config['data_name'])
    
    print("Seed: {}\n".format(config['seed']))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    train_loader, val_loader, test_loader = new_get_data(f, config)
    #train_plots(train_loader, 'train', 0)
    #train_plots(val_loader, 'val', 0)
    #raise
    
    ################################################################
    # training and evaluation
    ################################################################
    
    if(config['return_text']):
        #_data, _, _, _, _ = next(iter(val_loader))
        _data, _, _, _ = next(iter(val_loader))
    else:
        _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if(isinstance(model, (OFormer2D, VisionTransformer))):
        print("\nUSING ONECYCLELER SCHEDULER\n")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
                                                        steps_per_epoch=len(train_loader), epochs=config['epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
    
    loss_fn = nn.L1Loss(reduction="mean")
    loss_val_min = np.infty
    
    start_epoch = 0

    # TODO: Model restarting
    #if continue_training:
    #    print('Restoring model (that is the network\'s weights) from file...')
    #    checkpoint = torch.load(model_path, map_location=device)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    model.to(device)
    #    model.train()
    #    
    #    # Load optimizer state dict
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    for state in optimizer.state.values():
    #        for k, v in state.items():
    #            if isinstance(v, torch.Tensor):
    #                state[k] = v.to(device)
    #                
    #    start_epoch = checkpoint['epoch']
    #    loss_val_min = checkpoint['loss']
    
    train_l2s, val_l2s = [], []
    for ep in tqdm(range(start_epoch, config['epochs'])):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        #for bn, (xx, yy, grid, tokens, t) in enumerate(train_loader):
        for bn, (xx, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
            
            # Put data on correct device
            if(config['train_style'] == 'next_step'):
                random_step = random.randint(config['initial_step'], config['sim_time'])
                yy = xx[:, random_step].unsqueeze(-1).to(device).float()
                xx = xx[:, random_step-config['initial_step']:random_step].permute(0,2,3,1).to(device).float()
            else:
                yy = xx[:, config['sim_time']].unsqueeze(-1).to(device).float()
                xx = xx[:, :config['initial_step']].permute(0, 2, 3, 1).to(device).float()
            #xx = xx.to(device).float()
            #yy = yy.to(device).float()
            grid = grid.to(device).float()
            
            # Each model handles input differnetly
            if(isinstance(model, (FNO2d, OFormer2D))):
                if(navier_stokes):
                    x = torch.swapaxes(xx, 1, 3)
                    x = torch.swapaxes(x, 1, 2)
                else:
                    x = xx
                im, loss = model.get_loss(x, yy[...,0], grid, loss_fn)
            elif(isinstance(model, DeepONet2D)):
                #if(navier_stokes):
                #    x = torch.swapaxes(xx, 1, 3)
                #    x = torch.swapaxes(x, 1, 2)
                #else:
                #    x = xx
                im = model(xx, grid)
                loss = loss_fn(yy[...,0], im)
            elif(isinstance(model, CLIPFNO2d)):
                if(config['coeff']):
                    nu = coeffs['nu'].unsqueeze(-1)
                    ax = coeffs['ax'].unsqueeze(-1)
                    ay = coeffs['ay'].unsqueeze(-1)
                    cx = coeffs['cx'].unsqueeze(-1)
                    cy = coeffs['cy'].unsqueeze(-1)
                    coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(xx.shape[0], xx.shape[1], xx.shape[2], 5)
                    inp = torch.cat((xx, coeff), dim=-1)
                else:
                    inp = xx
                im, loss = model.get_loss(inp, yy[...,0], grid, sentence_embeddings, loss_fn)
                #print(coeff.shape, xx.shape)
            elif(isinstance(model, VisionTransformer)):
                if(config['coeff']):
                    nu = coeffs['nu'].unsqueeze(-1)
                    ax = coeffs['ax'].unsqueeze(-1)
                    ay = coeffs['ay'].unsqueeze(-1)
                    cx = coeffs['cx'].unsqueeze(-1)
                    cy = coeffs['cy'].unsqueeze(-1)
                    coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(xx.shape[0], xx.shape[1], xx.shape[2], 5)
                    inp = torch.cat((xx, grid, coeff), dim=-1).permute(0,3,1,2)
                else:
                    inp = torch.cat((xx, grid), dim=-1).permute(0,3,1,2)
                #print(coeff.shape, xx.shape)
                #raise
                #im = model(torch.cat((xx, grid), dim=-1).permute(0,3,1,2))
                im = model(inp)
                loss = loss_fn(yy, im)

            # Guarantees we're able to plot at least a few from first batch
            if(bn == 0):
                y_train_true = yy[...,0].clone()
                y_train_pred = im.clone()

            train_l2_step += loss.item()
            train_l2_full += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(isinstance(model, OFormer2D)):
                scheduler.step()

        train_l2s.append(train_l2_full/(bn+1))
        bn1 = bn

        if ep % config['validate'] == 0:
            val_l2_step = 0
            val_l2_full = 0
            model.eval()
            with torch.no_grad():
                #for bn, (xx, yy, grid, tokens, t) in enumerate(val_loader):
                for bn, (xx, grid, coeffs, sentence_embeddings) in enumerate(val_loader):

                    # Put data on correct device
                    if(config['train_style'] == 'next_step'):
                        random_step = random.randint(config['initial_step'], config['sim_time'])
                        yy = xx[:, random_step].unsqueeze(-1).to(device).float()
                        xx = xx[:, random_step-config['initial_step']:random_step].permute(0,2,3,1).to(device).float()
                    else:
                        yy = xx[:, config['sim_time']].unsqueeze(-1).to(device).float()
                        xx = xx[:, :config['initial_step']].permute(0, 2, 3, 1).to(device).float()
                    #xx = xx.to(device).float()
                    #yy = yy.to(device).float()
                    grid = grid.to(device).float()
                    
                    # Each model handles input differnetly
                    if(isinstance(model, (FNO2d, OFormer2D))):
                        if(navier_stokes):
                            x = torch.swapaxes(xx, 1, 3)
                            x = torch.swapaxes(x, 1, 2)
                        else:
                            x = xx
                        im, loss = model.get_loss(x, yy[...,0], grid, loss_fn)
                    elif(isinstance(model, DeepONet2D)):
                        #if(navier_stokes):
                        #    x = torch.swapaxes(xx, 1, 3)
                        #    x = torch.swapaxes(x, 1, 2)
                        #else:
                        #    x = xx
                        im = model(xx, grid)
                        loss = loss_fn(yy[...,0], im)
                    elif(isinstance(model, CLIPFNO2d)):
                        if(config['coeff']):
                            nu = coeffs['nu'].unsqueeze(-1)
                            ax = coeffs['ax'].unsqueeze(-1)
                            ay = coeffs['ay'].unsqueeze(-1)
                            cx = coeffs['cx'].unsqueeze(-1)
                            cy = coeffs['cy'].unsqueeze(-1)
                            coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(xx.shape[0], xx.shape[1], xx.shape[2], 5)
                            inp = torch.cat((xx, coeff), dim=-1)
                        else:
                            inp = xx
                        im, loss = model.get_loss(inp, yy[...,0], grid, sentence_embeddings, loss_fn)
                    elif(isinstance(model, VisionTransformer)):
                        if(config['coeff']):
                            nu = coeffs['nu'].unsqueeze(-1)
                            ax = coeffs['ax'].unsqueeze(-1)
                            ay = coeffs['ay'].unsqueeze(-1)
                            cx = coeffs['cx'].unsqueeze(-1)
                            cy = coeffs['cy'].unsqueeze(-1)
                            coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(xx.shape[0], xx.shape[1], xx.shape[2], 5)
                            inp = torch.cat((xx, grid, coeff), dim=-1).permute(0,3,1,2)
                        else:
                            inp = torch.cat((xx, grid), dim=-1).permute(0,3,1,2)

                        #im = model(torch.cat((xx, grid), dim=-1).permute(0,3,1,2))
                        im = model(inp)
                        loss = loss_fn(yy, im)

                    # Guarantees we're able to plot at least a few from first batch
                    if(bn == 0):
                        y_val_true = yy[...,0].clone()
                        y_val_pred = im.clone()

                    val_l2_step += loss.item()
                    val_l2_full += loss.item()
                
                if  val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_val_min
                        }, model_path)
        val_l2s.append(val_l2_full/(bn+1))
                
            
        t2 = default_timer()
        if(not isinstance(model, OFormer2D)):
            scheduler.step()

        if(ep%config['log_freq'] == 0):
            print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
                .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
            np.save("./{}/train_l2s_{}.npy".format(path, seed), train_l2s)
            np.save("./{}/val_l2s_{}.npy".format(path, seed), val_l2s)

        if(ep%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)

    # Make sure to capture last
    print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
          .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
    np.save("./{}/train_l2s_{}.npy".format(path, seed), train_l2s)
    np.save("./{}/val_l2s_{}.npy".format(path, seed), val_l2s)
    if(len(y_train_true) >= 4): 
        progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)

    test_vals = []
    test_value = evaluate(test_loader, model, loss_fn, navier_stokes, config=config)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, model, loss_fn, navier_stokes, config=config)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)

            
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
    train_args = config['args']
    train_args['model_name'] = model_name
    device = train_args['device']#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefix = train_args['data_name'].split("_")[0] + "_" + train_args['train_style']
    prefix += '_coeff' if(train_args['coeff']) else ''
    if('electric' in train_args['data_name']):
        prefix = 'electric_' + prefix
    os.makedirs("{}{}_{}/{}_{}".format(train_args['results_dir'], train_args['num_samples'], train_args['pretraining_num_samples'], model_name, prefix), exist_ok=True)
    shutil.copy("./configs/2d_{}_config.yaml".format(model_name),
                "{}{}_{}/{}_{}/2d_{}_config.yaml".format(train_args['results_dir'], train_args['num_samples'], train_args['pretraining_num_samples'], model_name, prefix, model_name))
    shutil.copy("./plot_progress.py", "{}{}_{}/{}_{}/plot_progress.py".format(train_args['results_dir'], train_args['num_samples'], train_args['pretraining_num_samples'], model_name, prefix))
    shutil.copy("./pretrain_plot_progress.py", "{}{}_{}/{}_{}/pretrain_plot_progress.py".format(train_args['results_dir'], train_args['num_samples'], train_args['pretraining_num_samples'], model_name, prefix))

    for seed in range(train_args.pop('num_seeds')):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_args['seed'] = seed

        #model = get_model(model_name, train_args)
        model = run_pretraining(model_name, train_args, prefix)
        run_training(model, train_args, prefix)
    print("Done.")

