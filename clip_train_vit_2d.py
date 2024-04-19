import torch
import torch.nn as nn
import yaml
import h5py
from utils import TransformerOperatorDataset2D, ElectricTransformerOperatorDataset2D
from anthony_data_handling import PDEDataset2D
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import shutil

from models.pitt import StandardPhysicsInformedTokenTransformer2D
from models.pitt import PhysicsInformedTokenTransformer2D
from models.pitt import CLIPPhysicsInformedTokenTransformer2D
from models.pitt import CLIPTransformer2D
from models.vit import CLIPVisionTransformer

from models.oformer import OFormer2D, SpatialTemporalEncoder2D, STDecoder2D, PointWiseDecoder2D
from models.deeponet import DeepONet2D
from models.fno import FNO2d

import sys

device = 'cuda' if(torch.cuda.is_available()) else 'cpu'


def custom_collate(batch):
    x0 = torch.empty((len(batch), batch[0][0].shape[0]))
    y = torch.empty((len(batch), batch[0][1].shape[0], 1))
    grid = torch.empty((len(batch), batch[0][2].shape[0]))
    tokens = torch.empty((len(batch), batch[0][3].shape[0]))
    forcing = []
    time = torch.empty(len(batch))
    for idx, b in enumerate(batch):
        x0[idx] = b[0]
        y[idx] = b[1]
        grid[idx] = b[2]
        tokens[idx] = b[3]
        forcing.append(b[4])
        time[idx] = b[5]
    return x0, y, grid, tokens, forcing, time


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

    ax[0][0].set_ylabel("VALIDATION SET TRUE")
    ax[1][0].set_ylabel("VALIDATION SET PRED")
    fname = str(ep)
    plt.tight_layout()
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
            ax.plot(v.reshape(100,).detach().cpu())
            ax.plot(preds[0][idx].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./val_2/{}_{}.png".format(seed, fname))
            plt.close()

            im_num += 1


def evaluate(test_loader, transformer, loss_fn, config=None):
    #src_mask = generate_square_subsequent_mask(640).cuda()
    with torch.no_grad():
        transformer.eval()
        test_loss = 0
        #for bn, (x0, y, grid, tokens, t, sentence_embeddings) in enumerate(test_loader):
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(test_loader):
            # Forward pass: compute predictions by passing the input sequence
            # through the transformer.
            y = x0[:, config['sim_time']].unsqueeze(-1)
            x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)

            # Put data on correct device
            x0 = x0.to(device).float()
            y = y.to(device).float()
            grid = grid.to(device).float()

            # Rearrange data
            if(config['coeff']):
                nu = coeffs['nu'].unsqueeze(-1)
                ax = coeffs['ax'].unsqueeze(-1)
                ay = coeffs['ay'].unsqueeze(-1)
                cx = coeffs['cx'].unsqueeze(-1)
                cy = coeffs['cy'].unsqueeze(-1)
                coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],     x0.shape[2], 5)
                inp = torch.cat((x0, grid, coeff), dim=-1).permute(0,3,1,2)
            else:
                inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)
            #y_pred = transformer(torch.cat((x0, grid), dim=-1).permute(0,3,1,2), sentence_embeddings, True)
            y_pred = transformer(inp, sentence_embeddings)
            y = y.to(device=device)
    
            # Compute the loss.
            test_loss += loss_fn(y_pred, y).item()

    return test_loss/(bn+1)


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


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
            clip=config['clip'],
            downsample=config['downsample']
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
            clip=config['clip'],
            downsample=config['downsample']
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
            clip=config['clip'],
            downsample=config['downsample']
    )
    batch_size = config['pretraining_batch_size'] if(pretraining) else config['batch_size']
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, generator=torch.Generator(device='cuda'),
                                               num_workers=config['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, generator=torch.Generator(device='cuda'),
                                             num_workers=config['num_workers'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             num_workers=config['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def get_neural_operator(model_name, config, temporal=False):
    if(model_name == "fno"):
        model = FNO2d(config['num_channels'], config['modes1'], config['modes2'], config['width'], config['initial_step'],
                      config['dropout'])
    elif(model_name == "unet"):
        model = UNet2d(in_channels=config['initial_step'], init_features=config['init_features'], dropout=config['dropout'])
    elif(model_name == "oformer"):
        if(temporal):
            encoder = SpatialTemporalEncoder2D(input_channels=config['input_channels']-1, in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], heads=config['heads'])
        else:
            encoder = SpatialTemporalEncoder2D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], heads=config['heads'])
                            #, dropout=config['dropout'],
                            #res=config['enc_res'])
        decoder = PointWiseDecoder2D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     propagator_depth=config['decoder_depth'], scale=config['scale'], out_steps=1)
        model = OFormer2D(encoder, decoder, num_x=config['num_x'], num_y=config['num_y'])
    elif(model_name == 'deeponet'):
        if(temporal):
            model = DeepONet2D(layer_sizes_branch=config['temporal_branch_net'], layer_sizes_trunk=config['trunk_net'],
                                activation=config['activation'], kernel_initializer=config['kernel_initializer'])
        else:
            model = DeepONet2D(layer_sizes_branch=config['branch_net'], layer_sizes_trunk=config['trunk_net'],
                                activation=config['activation'], kernel_initializer=config['kernel_initializer'])

    model.to(device)
    return model


def get_transformer(model_name, config):
    # Create the transformer model.
    if(config['embedding'] == "standard"):
        print("\n USING STANDARD EMBEDDING")
        neural_operator = get_neural_operator(config['neural_operator'], config)
        transformer = StandardPhysicsInformedTokenTransformer2D(100, config['hidden'], config['layers'], config['heads'],
                                        output_dim1=config['num_x'], output_dim2=config['num_y'], dropout=config['dropout'],
                                        neural_operator=neural_operator).to(device=device)
    elif(config['embedding'] == "novel"):
        print("\nUSING NOVEL EMBEDDING")
        neural_operator = get_neural_operator(config['neural_operator'], config)
        transformer = PhysicsInformedTokenTransformer2D(100, config['hidden'], config['layers'], config['heads'],
                                        output_dim1=config['num_x'], output_dim2=config['num_y'], dropout=config['dropout'],
                                        neural_operator=neural_operator).to(device=device)
    elif(config['embedding'] == "clip"):
        print("\nUSING CLIP EMBEDDING")
        neural_operator = get_neural_operator(config['neural_operator'], config)
        temporal_neural_operator = get_neural_operator(config['neural_operator'], config, temporal=True)
        #transformer = CLIPPhysicsInformedTokenTransformer2D(100, config['hidden'], config['layers'], config['heads'],
        transformer = CLIPTransformer2D(100, config['hidden'], config['layers'], config['heads'],
                                        output_dim1=config['num_x'], output_dim2=config['num_y'], dropout=config['dropout'],
                                        neural_operator=neural_operator, temporal_neural_operator=temporal_neural_operator,
                                        latent_dim=config['latent_dim']).to(device=device)
    elif(model_name == 'vit'):
        transformer = CLIPVisionTransformer(
                   img_size=config['img_size'],
                   patch_size=config['patch_size'],
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

    else:
        raise ValueError("Invalid embedding choice.")
    return transformer


def run_pretraining(config, prefix):
    #prefix = config['data_name'].split("_")[0]
    path = "{}{}_{}/{}_vit".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
    model_name = 'pretraining_vit' + "_{}.pt".format(seed)
    model_path = path + "/" + model_name

    # Create the transformer model.
    transformer = get_transformer('vit', config)
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
        #_data, _, _, _, _, _ = next(iter(val_loader))
        _data, _, _, _ = next(iter(val_loader))
    else:
        _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)

    
    # Use Adam as the optimizer.
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['pretraining_learning_rate'],
                                 weight_decay=config['pretraining_weight_decay'])
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
                inp = torch.cat((x0, grid, coeff), dim=-1).permute(0,3,1,2)
            else:
                inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)
            #y_pred = transformer(torch.cat((x0, grid), dim=-1).permute(0,3,1,2), sentence_embeddings, True)
            y_pred = transformer(inp, sentence_embeddings, True)

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
                    inp = torch.cat((x0, grid, coeff), dim=-1).permute(0,3,1,2)
                else:
                    inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)
                #y_pred = transformer(torch.cat((x0, grid), dim=-1).permute(0,3,1,2), sentence_embeddings, True)
                y_pred = transformer(inp, sentence_embeddings, True)

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
    transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, transformer, loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("{}{}_{}/{}_vit/pretraining_test_vals_{}.npy".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix, seed), test_vals)

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
                    inp = torch.cat((x0, grid, coeff), dim=-1).permute(0,3,1,2)
                else:
                    inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)
                emb = transformer(inp, sentence_embeddings, False, True)

                embs.append(emb)
                all_coeffs.append(torch.vstack(list(coeffs.values())).transpose(0,1))

        all_embs = torch.cat(embs, dim=0)
        all_coeffs = torch.cat(all_coeffs, dim=0)
        print(all_embs.shape)
        print(all_coeffs.shape)
        np.save("./{}/pretraining_embeddings_{}.npy".format(path, seed), all_embs.cpu().numpy())
        np.save("./{}/pretraining_coeffs_{}.npy".format(path, seed), all_coeffs.cpu().numpy())

    return transformer


def run_training(transformer, config, prefix):
    #prefix = config['data_name'].split("_")[0]
    path = "{}{}_{}/{}_vit".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
    model_name = 'vit' + "_{}.pt".format(seed)
    model_path = path + "/" + model_name

    # Create the transformer model.

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    train_loader, val_loader, test_loader = new_get_data(f, config)

    ################################################################
    # training and evaluation
    ################################################################

    if(config['return_text']):
        #_data, _, _, _, _, _ = next(iter(val_loader))
        _data, _, _, _ = next(iter(val_loader))
    else:
        _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)


    # Use Adam as the optimizer.
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
            steps_per_epoch=len(train_loader), epochs=config['epochs'])

    # Use mean squared error as the loss function.
    loss_fn = nn.L1Loss(reduction='mean')

    # Train the transformer for the specified number of epochs.
    train_losses = []
    val_losses = []
    loss_val_min = np.infty
    #src_mask = generate_square_subsequent_mask(640).cuda()
    lrs = []
    shift = 0
    print("\nTRAINING...")
    for epoch in tqdm(range(config['epochs'])):
        # Iterate over the training dataset.
        train_loss = 0
        times = []
        max_val = 0
        transformer.train()
        #for bn, (x0, y, grid, tokens, t, sentence_embeddings) in enumerate(train_loader):
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
            start = time.time()

            y = x0[:, config['sim_time']].unsqueeze(-1)
            x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)

            # Put data on correct device
            x0 = x0.to(device).float()
            y = y.to(device).float()
            #tokens = tokens.to(device).float()
            #t = t.to(device).float()
            grid = grid.to(device).float()

            # Rearrange data
            #if(not('electric' in config['data_name'])):
            #    x0 = torch.swapaxes(x0, 1, 3)
            #    x0 = torch.swapaxes(x0, 1, 2)

            # Forward pass
            #y_pred = transformer(grid, tokens, x0, t)
            #y_pred = transformer(grid, None, x0, None, clip=False)
            if(config['coeff']):
                nu = coeffs['nu'].unsqueeze(-1)
                ax = coeffs['ax'].unsqueeze(-1)
                ay = coeffs['ay'].unsqueeze(-1)
                cx = coeffs['cx'].unsqueeze(-1)
                cy = coeffs['cy'].unsqueeze(-1)
                coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],     x0.shape[2], 5)
                inp = torch.cat((x0, grid, coeff), dim=-1).permute(0,3,1,2)
            else:
                inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)
            #y_pred = transformer(torch.cat((x0, grid), dim=-1).permute(0,3,1,2), sentence_embeddings, True)
            y_pred = transformer(inp, sentence_embeddings)

            y = y.to(device=device)#.cuda()

            # Compute the loss.
            loss = loss_fn(y_pred, y)

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

        with torch.no_grad():
            transformer.eval()
            val_loss = 0
            all_val_preds = []
            #for bn, (x0, y, grid, tokens, t, sentence_embeddings) in enumerate(val_loader):
            for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(val_loader):
                # Forward pass: compute predictions by passing the input sequence
                # through the transformer.
                # Put data on correct device
                y = x0[:, config['sim_time']].unsqueeze(-1)
                x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)
                x0 = x0.to(device).float()
                y = y.to(device).float()
                #tokens = tokens.to(device).float()
                #t = t.to(device).float()
                grid = grid.to(device).float()

                # Rearrange data
                #if(not('electric' in config['data_name'])):
                #    x0 = torch.swapaxes(x0, 1, 3)
                #    x0 = torch.swapaxes(x0, 1, 2)

                #y_pred = transformer(grid, tokens, x0, t)
                #y_pred = transformer(grid, None, x0, None, clip=False)
                if(config['coeff']):
                    nu = coeffs['nu'].unsqueeze(-1)
                    ax = coeffs['ax'].unsqueeze(-1)
                    ay = coeffs['ay'].unsqueeze(-1)
                    cx = coeffs['cx'].unsqueeze(-1)
                    cy = coeffs['cy'].unsqueeze(-1)
                    coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],     x0.shape[2], 5)
                    inp = torch.cat((x0, grid, coeff), dim=-1).permute(0,3,1,2)
                else:
                    inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)
                y_pred = transformer(inp, sentence_embeddings)
                y = y.to(device=device)#.cuda()
                if(bn == 0):
                    y_val_true = y.clone()
                    y_val_pred = y_pred.clone()
                all_val_preds.append(y_pred.detach())

                # Compute the loss.
                val_loss += loss_fn(y_pred, y).item()

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
            np.save("./{}/train_l2s_{}.npy".format(path, seed), train_losses)
            np.save("./{}/val_l2s_{}.npy".format(path, seed), val_losses)
            np.save("./{}/lrs_{}.npy".format(path, seed), lrs)
            print(f"Epoch {epoch+1}: loss = {train_loss:.6f}\t val loss = {val_loss:.6f}")

        if(epoch%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)

    progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)
    #val_plots(epoch, val_loader, all_val_preds, seed=seed)

    test_vals = []
    test_value = evaluate(test_loader, transformer, loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, transformer, loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("{}{}_{}/{}_vit/test_vals_{}.npy".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix, seed), test_vals)


if __name__ == '__main__':
    # Create a transformer with an input dimension of 10, a hidden dimension
    # of 20, 2 transformer layers, and 8 attention heads.

    # Load config
    #raise
    with open("./configs/2d_vit_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    prefix = train_args['data_name'].split("_")[0] + "_" + train_args['train_style'] + "_" + train_args['embedding']
    prefix += "_coeff" if(train_args['coeff']) else ""
    if('electric' in train_args['data_name']):
        prefix = "electric_" + prefix
    train_args['prefix'] = prefix
    os.makedirs("{}{}_{}/{}_vit".format(train_args['results_dir'], train_args['num_samples'], train_args['pretraining_num_samples'], prefix),
                exist_ok=True)
    shutil.copy("./configs/2d_vit_config.yaml",
                "{}{}_{}/{}_vit/2d_vit_config.yaml".format(train_args['results_dir'], train_args['num_samples'], train_args['pretraining_num_samples'],
                prefix))
    shutil.copy("./plot_progress.py", "{}{}_{}/{}_vit/plot_progress.py".format(train_args['results_dir'], train_args['num_samples'], train_args['pretraining_num_samples'], 
                prefix))
    shutil.copy("./pretrain_plot_progress.py", "{}{}_{}/{}_vit/pretrain_plot_progress.py".format(train_args['results_dir'], train_args['num_samples'], train_args['pretraining_num_samples'],
                prefix))


    for seed in range(train_args.pop('num_seeds')):
    #for seed in [0,1]:
    #for seed in [2,3]:
    #for seed in [4]:
        print("\nSEED: {}\n".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_args['seed'] = seed
        model = run_pretraining(train_args, prefix)
        run_training(model, train_args, prefix)
    
