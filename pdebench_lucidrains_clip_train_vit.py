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

from models.pitt import StandardPhysicsInformedTokenTransformer2D
from models.pitt import PhysicsInformedTokenTransformer2D
from models.pitt import CLIPPhysicsInformedTokenTransformer2D

#from models.vit import VisionTransformer
from models.lucidrains_vit import ViT, CLIPViT, LLMCLIPViT

from models.oformer import OFormer2D, SpatialTemporalEncoder2D, STDecoder2D, PointWiseDecoder2D
from models.deeponet import DeepONet2D
from models.fno import FNO2d
from models.transolver import EmbeddingTransolver

from helpers import get_data
from metrics import metric_func

import sys

device = 'cuda' if(torch.cuda.is_available()) else 'cpu'

DEBUG = True

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


def new_get_data(config, pretraining=False, subset='heat,adv,burger'):

    # Select specific file
    if(config['dataset'] == 'shallow_water'):
        filename = '2D_rdb_NA_NA.h5'
        extension = 'shallow-water'
    elif(config['dataset'] == 'diffusion_reaction'):
        filename = '2D_diff-react_NA_NA.h5'
        extension = 'diffusion-reaction'
    elif(config['dataset'] == 'all'):
        filenames = ['2D_rdb_NA_NA.h5', '2D_diff-react_NA_NA.h5']
        saved_folders = ["/home/cooperlorsung/pdebench_data/2D/{}".format(i) for i in ['shallow-water', 'diffusion-reaction']]
    else:
        raise NotImplementedError("Select shallow_water or diffusion_reaction for now.")

    if(config['dataset'] != 'all'):
        train_data = FNODatasetSingle(
                filename=filename,
                saved_folder="/home/cooperlorsung/pdebench_data/2D/{}".format(extension),
                initial_step=config['initial_step'],
                reduced_resolution=config['reduced_resolution'],
                reduced_resolution_t=config['reduced_resolution_t'],
                reduced_batch=1,
                if_test=False,
                test_ratio=0.1,
                num_samples_max=config['pretraining_num_samples'] if(pretraining) else config['num_samples'],

                clip=config['clip'],
                llm=config['llm'],
                coeff=config['coeff'],
                sentence=config['sentence'],
                qualitative=config['qualitative'],
        )
        val_data = FNODatasetSingle(
                filename=filename,
                saved_folder="/home/cooperlorsung/pdebench_data/2D/{}".format(extension),
                initial_step=1,
                reduced_resolution=config['reduced_resolution'],
                reduced_resolution_t=config['reduced_resolution_t'],
                reduced_batch=1,
                if_test=True,
                test_ratio=0.1,
                num_samples_max=config['num_samples'],

                clip=config['clip'],
                llm=config['llm'],
                coeff=config['coeff'],
                sentence=config['sentence'],
                qualitative=config['qualitative'],
        )
        test_data = FNODatasetSingle(
                filename=filename,
                saved_folder="/home/cooperlorsung/pdebench_data/2D/{}".format(extension),
                initial_step=1,
                reduced_resolution=config['reduced_resolution'],
                reduced_resolution_t=config['reduced_resolution_t'],
                reduced_batch=1,
                if_test=True,
                test_ratio=0.1,
                num_samples_max=config['num_samples'],

                clip=config['clip'],
                llm=config['llm'],
                coeff=config['coeff'],
                sentence=config['sentence'],
                qualitative=config['qualitative'],
        )
    else:
        print("\nTRAIN DATA")
        train_data = MultiDataset(
                filenames=filenames,
                saved_folders=saved_folders,
                initial_step=config['initial_step'],
                reduced_resolution=config['reduced_resolution'],
                reduced_resolution_t=config['reduced_resolution_t'],
                reduced_batch=1,
                if_test=False,
                test_ratio=0.1,
                num_samples_max=config['pretraining_num_samples'] if(pretraining) else config['num_samples'],

                clip=config['clip'],
                llm=config['llm'],
                coeff=config['coeff'],
                sentence=config['sentence'],
                qualitative=config['qualitative'],
        )
        print("\nVAL DATA")
        val_data = MultiDataset(
                filenames=filenames,
                saved_folders=saved_folders,
                initial_step=config['initial_step'],
                reduced_resolution=config['reduced_resolution'],
                reduced_resolution_t=config['reduced_resolution_t'],
                reduced_batch=1,
                if_test=True,
                test_ratio=0.1,
                num_samples_max=config['num_samples'],

                clip=config['clip'],
                llm=config['llm'],
                coeff=config['coeff'],
                sentence=config['sentence'],
                qualitative=config['qualitative'],
        )
        print("\nTEST DATA")
        test_data = MultiDataset(
                filenames=filenames,
                saved_folders=saved_folders,
                initial_step=config['initial_step'],
                reduced_resolution=config['reduced_resolution'],
                reduced_resolution_t=config['reduced_resolution_t'],
                reduced_batch=1,
                if_test=True,
                test_ratio=0.1,
                num_samples_max=config['num_samples'],

                clip=config['clip'],
                llm=config['llm'],
                coeff=config['coeff'],
                sentence=config['sentence'],
                qualitative=config['qualitative'],
        )
    batch_size = config['pretraining_batch_size'] if(pretraining) else config['batch_size']
    print("\nPRETRAINING: {}\n".format(pretraining))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, generator=torch.Generator(device='cuda'),
                                               num_workers=config['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, generator=torch.Generator(device='cuda'),
                                             num_workers=config['num_workers'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             num_workers=config['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def get_transformer(model_name, config):
    # Channels based on initial steps
    channels = config['initial_step']

    # Multiply by number of target variables
    channels *= 1 if(config['dataset'] == 'shallow_water') else 2 if(config['dataset'] == 'diffusion_reaction') else 4

    # Add 2 for grid
    channels += 2

    # Add more for coefficients
    channels += 5 if(config['coeff']) else 0

    # Add time channel if doing arbitraty_step training
    channels += 1 if(config['train_style'] == 'arbitrary_step') else 0

    # Out channels is number of target variables, only doing single step prediction for now
    out_channels = 1 if(config['dataset'] == 'shallow_water') else 2 if(config['dataset'] == 'diffusion_reaction') else 4

    # Create the transformer model.
    if(config['sentence']):
        print("USING SENTENCE CLIP VISION TRANSFORMER WITH: {}\n".format(config['llm']))
        transformer = LLMCLIPViT(
                   image_size=config['img_size'],
                   patch_size=config['patch_size'],
                   dim=config['dim'],
                   depth=config['depth'],
                   heads=config['heads'],
                   mlp_dim=config['mlp_dim'],
                   pool=config['pool'],
                   channels=channels,
                   out_channels=out_channels,
                   dim_head=config['dim_head'],
                   dropout=config['dropout'],
                   emb_dropout=config['emb_dropout'],
                   llm=config['llm'],
        ).to(device)

    elif(model_name == 'vit'):
        print("USING CLIP VISION TRANSFORMER WITH: {}\n".format(config['llm']))
        transformer = CLIPViT(
                   image_size=config['img_size'],
                   patch_size=config['patch_size'],
                   dim=config['dim'],
                   depth=config['depth'],
                   heads=config['heads'],
                   mlp_dim=config['mlp_dim'],
                   pool=config['pool'],
                   channels=channels,
                   out_channels=out_channels,
                   dim_head=config['dim_head'],
                   dropout=config['dropout'],
                   emb_dropout=config['emb_dropout'],
                   llm=config['llm'],
        ).to(device)

#    elif(model_name == 'transolver'):
#        transformer = EmbeddingTransolver(
#                space_dim=channels-2, # Spatial - grid dimensions
#                fun_dim=2,            # grid dimension
#                llm=config['llm']
#        ).to(device)
    elif(model_name == 'transolver'):
        transformer = EmbeddingTransolver(
                space_dim=channels-2,      # Spatial - grid dimensions
                fun_dim=2,                 # grid dimension
                out_dim = out_channels,    # Number of output channels varies based on dataset
                H=config['img_size'],
                W=config['img_size'],
                llm=config['llm'],
        ).to(device)


    else:
        raise ValueError("Invalid model choice.")
    return transformer


def get_pretraining_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn, times=None):
    # Select data for input and target
    if(config['train_style'] == 'next_step'):
        steps = torch.Tensor(np.random.choice(range(config['initial_step'], config['sim_time']),
                             x0.shape[0])).long()
        try:
            y = torch.cat([x0[idx,i][None,None] for idx, i in enumerate(steps)], dim=0)
            x0 = torch.cat([x0[idx,i-config['initial_step']:i][None] for idx, i in enumerate(steps    )], dim=0)
        except IndexError:
            print(steps)
            print(x0.shape)
            raise

        y = y.permute(0,2,3,1)
        x0 = x0.permute(0,2,3,1) if(len(x0.shape) == 4) else x0.unsqueeze(-1)
    elif(config['train_style'] == 'fixed_future'):
        y = x0[:, config['sim_time']]
        x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)
    elif(config['train_style'] == 'arbitrary_step'):
        # Generate random slices
        steps = torch.Tensor(np.random.choice(range(config['initial_step'], config['sim_time']+1),
                             x0.shape[0])).long()
        y = torch.cat([x0[idx,i][None,None] for idx, i in enumerate(steps)], dim=0)
        t = torch.cat([times[i][None] for idx, i in enumerate(steps)], dim=0)

        # Use initial condition and stack target time
        x0 = x0[:,:config['initial_step']]
        if(times is None):
            raise ValueError("Need target times to stack with data.")
        x0 = torch.cat((x0, t[:,None,None,None].broadcast_to(x0.shape[0], 1, x0.shape[2], x0.shape[3])), axis=1)

        # Reorder dimensions
        y = y.permute(0,2,3,1)
        x0 = x0.permute(0,2,3,1) if(len(x0.shape) == 4) else x0.unsqueeze(-1)

    # Put data on correct device
    x0 = x0.to(device).float()
    y = y.to(device).float()
    grid = grid.to(device).float()
    if(not isinstance(sentence_embeddings, tuple)):
        sentence_embeddings = sentence_embeddings.to(device).float()

    if(config['coeff']):

        # Get all coefficients
        nu = coeffs['nu'].unsqueeze(-1)
        ax = coeffs['ax'].unsqueeze(-1)
        ay = coeffs['ay'].unsqueeze(-1)
        cx = coeffs['cx'].unsqueeze(-1)
        cy = coeffs['cy'].unsqueeze(-1)

        # Stack coefficients together
        coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],
                                                                                               x0.shape[2], 5)
        # Stack data and coefficients (TODO: Do we really need grid information for ViT?)
        inp = torch.cat((x0, grid, coeff), dim=-1).permute(0,3,1,2)
    else:
        inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)

    # Forward pass
    y_pred = transformer(inp, sentence_embeddings, clip=True)
    labels = generate_pretraining_labels(config, coeffs, y_pred)
    loss = loss_fn(y_pred, labels)
    return loss


def save_embeddings(config, path, transformer, loader, train=True, seed=0):
    embs = []
    all_coeffs = []
    all_sim_mats = []
    transformer.eval()
    with torch.no_grad():
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(loader):
            if(config['train_style'] == 'next_step'):
                raise ValueError("Not Implemented Yet.")
            elif(config['train_style'] == 'fixed_future'):
                x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)
            elif(config['train_style'] == 'arbitrary_step'):
                # Generate random slices
                steps = torch.Tensor(np.random.choice(range(config['initial_step'], config['sim_time']+1),
                                     x0.shape[0])).long()
                y = torch.cat([x0[idx,i][None,None] for idx, i in enumerate(steps)], dim=0)
                t = torch.cat([times[i][None] for idx, i in enumerate(steps)], dim=0)

                # Use initial condition and stack target time
                x0 = x0[:,:config['initial_step']]
                if(times is None):
                    raise ValueError("Need target times to stack with data.")
                x0 = torch.cat((x0, t[:,None,None,None].broadcast_to(x0.shape[0], 1, x0.shape[2], x0.shape[3])), axis=1)

                # Reorder dimensions
                #y = y.permute(0,2,3,1)
                x0 = x0.permute(0,2,3,1) if(len(x0.shape) == 4) else x0.unsqueeze(-1)

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

            emb, sim_mat = transformer(inp, sentence_embeddings, False, True) # Get stacked embeddings

            embs.append(emb)
            all_coeffs.append(torch.vstack(list(coeffs.values())).transpose(0,1))

            # Need to find a way to exclude last batch
            if(sim_mat.shape[0] == config['pretraining_batch_size']):
                all_sim_mats.append(sim_mat.unsqueeze(0))

    all_embs = torch.cat(embs, dim=0)
    all_coeffs = torch.cat(all_coeffs, dim=0)
    all_sim_mats = torch.cat(all_sim_mats, dim=0)

    split = "train" if(train) else "val"
    np.save("./{}/pretraining_{}_embeddings_{}.npy".format(path, split, seed), all_embs.cpu().numpy())
    np.save("./{}/pretraining_{}_coeffs_{}.npy".format(path, split, seed), all_coeffs.cpu().numpy())
    np.save("./{}/pretraining_{}_sim_mats_{}.npy".format(path, split, seed), all_sim_mats.cpu().numpy())


def run_pretraining(config, prefix):
    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    pretrained_path = "{}{}/{}".format(config['pretrained_model_path'], config['pretraining_num_samples'],
                                       prefix)

    model_name = 'pretraining' + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    pretrained_model_path = pretrained_path + "/" + model_name

    # Create the transformer model.
    transformer = get_transformer('vit', config)
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

    _data, _, _, _ = next(iter(train_loader))
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
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
            start = time.time()
            loss = get_pretraining_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn,
                                        times=train_loader.dataset.t)

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
            for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(val_loader):
                loss = get_pretraining_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn,
                                            times=val_loader.dataset.t)
                val_loss += loss.item()

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
        if(config['train_style'] == 'fixed_future'):
            save_embeddings(config, path, transformer, train_loader, seed=seed, train=True)
            save_embeddings(config, path, transformer, val_loader, seed=seed, train=False)

    return transformer, pretrained_model_path


def get_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn, times=None, evaluate=False):

    # Select data for input and target
    if(config['train_style'] == 'next_step'):
        steps = torch.Tensor(np.random.choice(range(config['initial_step'], config['sim_time']), x0.shape[0])).long()
        #steps = torch.Tensor(np.random.choice(range(config['initial_step'], len(times)), x0.shape[0])).long()
        try:
            y = torch.cat([x0[idx,:,:,i][None] for idx, i in enumerate(steps)], dim=0)
            #x0 = torch.cat([x0[idx,:,:,i-config['initial_step']:i][None] for idx, i in enumerate(steps)], dim=0)[...,0,:]
            x0 = torch.cat([x0[idx,:,:,i-config['initial_step']:i][None] for idx, i in enumerate(steps)], dim=0).flatten(3,4)
        except IndexError:
            print(steps)
            print(x0.shape)
            raise

    elif(config['train_style'] == 'fixed_future'):
        y = x0[:, config['sim_time']].unsqueeze(-1)
        x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)
    elif(config['train_style'] == 'arbitrary_step'):
        # Generate random slices
        steps = torch.Tensor(np.random.choice(range(config['initial_step'], config['sim_time']+1),
                             x0.shape[0])).long()
        y = torch.cat([x0[idx,i][None,None] for idx, i in enumerate(steps)], dim=0)
        t = torch.cat([times[i][None] for idx, i in enumerate(steps)], dim=0)
    
        # Use initial condition and stack target time
        x0 = x0[:,:config['initial_step']]
        if(times is None):
            raise ValueError("Need target times to stack with data.")
        x0 = torch.cat((x0, t[:,None,None,None].broadcast_to(x0.shape[0], 1, x0.shape[2], x0.shape[3])), axis=1)

        # Reorder dimensions
        y = y.permute(0,2,3,1)
        x0 = x0.permute(0,2,3,1) if(len(x0.shape) == 4) else x0.unsqueeze(-1)


    # Put data on correct device
    x0 = x0.to(device).float()
    y = y.to(device).float()
    grid = grid.to(device).float()

    if(config['coeff']): # Stack coefficients
        nu = coeffs['nu'].unsqueeze(-1)
        ax = coeffs['ax'].unsqueeze(-1)
        ay = coeffs['ay'].unsqueeze(-1)
        cx = coeffs['cx'].unsqueeze(-1)
        cy = coeffs['cy'].unsqueeze(-1)
        coeff = torch.cat((nu,ax,ay,cx,cy), dim=-1).reshape(nu.shape[0], 1, 1, 5).broadcast_to(x0.shape[0], x0.shape[1],
                                                                                               x0.shape[2], 5)
        inp = torch.cat((x0, grid, coeff), dim=-1).permute(0,3,1,2)
    else:
        inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)

    if(isinstance(transformer, EmbeddingTransolver)):
        y_pred = transformer(grid, x0, sentence_embeddings)
    else:
        y_pred = transformer(inp, sentence_embeddings)[...,0].permute(0,2,3,1)

    y = y.to(device=device)

    # Compute the loss.
    if(evaluate):
        ch0 = not((y[...,-1,0] == 0).all())
        ch1 = not((y[...,-1,1] == 0).all())
        ch2 = not((y[...,-1,2] == 0).all())
        ch3 = not((y[...,-1,3] == 0).all())
        chans = [ch0, ch1, ch2, ch3]
        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(y_pred[...,chans], y[...,chans], if_mean=True,
                                                                           Lx=1., Ly=1., Lz=1.)
        return y_pred, y, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F
    else:
        loss = loss_fn(y_pred, y)
        return y_pred, y, loss


def evaluate(test_loader, transformer, loss_fn, config=None):
    #src_mask = generate_square_subsequent_mask(640).cuda()
    metrics = {'RMSE': [], 'nRMSE': [], 'CSV': [], 'Max': [], 'BD': [], 'F': []}
    with torch.no_grad():
        transformer.eval()
        test_loss = 0
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(test_loader):
            # Forward pass: compute predictions by passing the input sequence
            # through the transformer.
            #y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn, times=test_loader.dataset.tsteps_t)
            y_pred, y, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = get_loss(config, transformer, x0, grid, coeffs,
                                                                                       sentence_embeddings, loss_fn,
                                                                                       times=test_loader.dataset.tsteps_t,
                                                                                       evaluate=True)
            #test_loss += loss.item()
            test_loss += err_nRMSE.item()

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
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(test_loader):
            #y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn,
            #                           times=test_loader.dataset.tsteps_t, evaluate=True)
            y_pred, y, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = get_loss(config, transformer, x0, grid, coeffs,
                                                                                       sentence_embeddings, loss_fn,
                                                                                       times=test_loader.dataset.tsteps_t,
                                                                                       evaluate=True)
            #test_loss += loss.item()
            test_loss += err_nRMSE.item()

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


def as_rollout(test_loader, transformer, loss_fn, config, prefix, subset):
    all_y_preds, all_y_trues = [], []
    with torch.no_grad():
        transformer.eval()
        test_loss = 0

        # TODO: Loop over dataset not data loader
        #for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(test_loader):
        #for idx in tqdm(range(test_loader.dataset.u.shape[0])):
        for original_idx, idx in tqdm(enumerate(test_loader.dataset.indexes)):

            # Get steps for the entire trajectory
            steps = torch.Tensor([i for i in range(config['initial_step'], config['sim_time']+1)]).long()

            # Get data from single trajectory
            x0 = test_loader.dataset.u[idx].unsqueeze(0)
            grid = test_loader.dataset.x.repeat(len(steps), 1, 1, 1)

            # Need every slice but the first...
            y = torch.cat([x0[:,i][None] for idx, i in enumerate(steps)], dim=0).permute(0,2,3,1)

            # Need the initial step as many times as it takes to match the rest of the trajectory
            x0 = x0[:,:config['initial_step']].repeat(len(steps), 1, 1, 1)
            t = test_loader.dataset.t[config['initial_step']:]

            # Put data on correct device
            x0 = x0.to(device).float()
            y = y.to(device).float()
            grid = grid.to(device).float()

            # Stack target time
            x0 = torch.cat((x0, t[:,None,None,None].broadcast_to(x0.shape[0], 1, x0.shape[2], x0.shape[3])), axis=1)

            if(config['coeff']): # Stack coefficients
                nu = test_loader.dataset.nu[idx].unsqueeze(-1)
                ax = test_loader.dataset.ax[idx].unsqueeze(-1)
                ay = test_loader.dataset.ay[idx].unsqueeze(-1)
                cx = test_loader.dataset.cx[idx].unsqueeze(-1)
                cy = test_loader.dataset.cy[idx].unsqueeze(-1)

                coeff = torch.cat((nu,ax,ay,cx,cy), dim=0)[None,:,None,None].broadcast_to(x0.shape[0], 5, x0.shape[2], x0.shape[3])

                inp = torch.cat((x0, grid, coeff), dim=1)
            else:
                inp = torch.cat((x0, grid), dim=1)

            # Make prediction
            y_pred = transformer(inp)

            # Save data and pred
            all_y_preds.append(y_pred.unsqueeze(0))
            all_y_trues.append(y.unsqueeze(0))

    # Stack predictions and ground truth
    all_y_preds = torch.cat(all_y_preds, dim=0)
    all_y_trues = torch.cat(all_y_trues, dim=0)

    # Now in shape traj x time x space x channels
    mse = ((all_y_preds - all_y_trues)**2).mean(dim=(0,2))

    # Save relevant info
    #path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'],
    #                           config['pretraining_num_samples'], prefix)
    path = "{}{}/{}".format(config['results_dir'], config['num_samples'], prefix)

    if(subset != 'heat,adv,burger'):
        torch.save(mse, path+"/{}_{}_rollout_mse".format(seed, subset))
        torch.save(all_y_trues.cpu(), path+"/{}_{}_y_trues".format(seed, subset))
        torch.save(all_y_preds.cpu(), path+"/{}_{}_y_preds".format(seed, subset))
    else:
        torch.save(mse, path+"/{}_rollout_mse".format(seed))
        torch.save(all_y_trues.cpu(), path+"/{}_all_y_trues".format(seed))
        torch.save(all_y_preds.cpu(), path+"/{}_all_y_preds".format(seed))
    return test_loss/(idx+1)


def ar_rollout(test_loader, transformer, loss_fn, config, prefix, subset):
    all_y_preds, all_y_trues = [], []
    with torch.no_grad():
        transformer.eval()
        test_loss = 0

        # TODO: Loop over dataset not data loader
        print("Autoregressive rollout...")
        for idx in tqdm(range(test_loader.dataset.data.shape[0])):
            x0 = test_loader.dataset.data[idx][...,:config['initial_step'],:].unsqueeze(0).flatten(3,4)
            y = test_loader.dataset.data[idx][...,config['initial_step']:,:].unsqueeze(0)

            # Grid changes based on multi or single dataset.
            if(isinstance(test_loader.dataset, MultiDataset)):
                #TODO: This no longer supports unevenly sized data sets
                sample_idx = idx%(int(0.1*config['num_samples']))
                dset_idx = idx//int((0.1*config['num_samples']))

                #grid = test_loader.dataset.grids[idx//config['num_samples']].unsqueeze(0)

                grid = test_loader.dataset.grids[dset_idx].unsqueeze(0)
                sentence_embeddings = torch.Tensor(test_loader.dataset.dsets[dset_idx].sentence_embeddings[sample_idx]).unsqueeze(0)
            else:
                grid = test_loader.dataset.grid.unsqueeze(0)
                sentence_embeddings = test_loader.dataset.sentence_embeddings[idx].unsqueeze(0)

            if(len(x0.shape) == 5):
                x0 = x0[...,0,:]

            y_preds = []
            #print("\nY SHAPE: {}\n".format(y.shape))
            for i in range(y.shape[-2]):
                #print("STEP: {}".format(i))

                if(config['coeff']): # Stack coefficients
                    nu = test_loader.dataset.nu[idx].unsqueeze(-1)
                    ax = test_loader.dataset.ax[idx].unsqueeze(-1)
                    ay = test_loader.dataset.ay[idx].unsqueeze(-1)
                    cx = test_loader.dataset.cx[idx].unsqueeze(-1)
                    cy = test_loader.dataset.cy[idx].unsqueeze(-1)

                    coeff = torch.cat((nu,ax,ay,cx,cy), dim=0)[None,:,None,None].broadcast_to(x0.shape[0], 5,
                                                                                              x0.shape[2], x0.shape[3])

                    inp = torch.cat((x0, grid, coeff), dim=1)
                else:
                    inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2).cuda()

                #y_pred = transformer(inp, sentence_embeddings).permute(0,4,2,3,1)[0]

                ## Stack and slice on input
                ##x0 = torch.cat((x0, y_pred), dim=0)[config['initial_step']:]

                ## Take number of data channels and subtract grid channels
                #x0 = torch.cat((x0, y_pred), dim=-1)[...,-transformer.channels+2:]
                #raise
                # Make prediction
                if(isinstance(transformer, EmbeddingTransolver)):
                    y_pred = transformer(grid, x0, sentence_embeddings)
                    x0 = torch.cat((x0, y_pred), dim=-1)[...,-transformer.space_dim:]
                else:
                    y_pred = transformer(inp).permute(0,4,2,3,1)[0]
                    x0 = torch.cat((x0, y_pred), dim=-1)[...,-transformer.channels+2:]

                # Save preds
                y_preds.append(y_pred.unsqueeze(0))

            # Save data and preds
            all_y_preds.append(torch.cat(y_preds, dim=1))
            all_y_trues.append(y)

    # Stack predictions and ground truth
    all_y_preds = torch.cat(all_y_preds, dim=0).permute(0,2,3,1,4)
    all_y_trues = torch.cat(all_y_trues, dim=0)

    # Now in shape traj x time x space x channels
    mse = ((all_y_preds - all_y_trues)**2).mean(dim=(0,2))

    # Save relevant info
    #path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'],
    #                           config['pretraining_num_samples'], prefix)
    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)

    if(subset != 'heat,adv,burger'):
        torch.save(mse, path+"/rollouts/{}_{}_rollout_mse".format(seed, subset))
        torch.save(all_y_trues.cpu(), path+"/rollouts/{}_{}_y_trues".format(seed, subset))
        torch.save(all_y_preds.cpu(), path+"/rollouts/{}_{}_y_preds".format(seed, subset))
    else:
        torch.save(mse, path+"/rollouts/{}_rollout_mse".format(seed))
        torch.save(all_y_trues.cpu(), path+"/rollouts/{}_all_y_trues".format(seed))
        torch.save(all_y_preds.cpu(), path+"/rollouts/{}_all_y_preds".format(seed))
    return test_loss/(idx+1)


def run_training(transformer, config, prefix, seed, subset='heat,adv,burger'):
    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    model_name = 'vit' + "_{}.pt".format(seed)
    if(subset != 'heat,adv,burger'):
        model_name = subset + "_" + model_name
    model_path = path + "/" + model_name

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    #train_loader, val_loader, test_loader = new_get_data(config, subset=subset)
    train_loader, val_loader, test_loader = get_data(config)

    ################################################################
    # training and evaluation
    ################################################################

    _data, _, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)

    # Use Adam as the optimizer.
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
                                                    steps_per_epoch=len(train_loader), epochs=config['epochs'])

    # Use mean squared error as the loss function.
    #loss_fn = nn.L1Loss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='mean')
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
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
            start = time.time()
            y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn,
                                       times=train_loader.dataset.tsteps_t)

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
            for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(val_loader):
                # Forward pass: compute predictions by passing the input sequence
                y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn,
                                           times=train_loader.dataset.tsteps_t)
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
            if(subset != 'heat,adv,burger'):
                np.save("./{}/{}_train_l2s_{}.npy".format(path, subset, seed), train_losses)
                np.save("./{}/{}_val_l2s_{}.npy".format(path, subset, seed), val_losses)
                np.save("./{}/{}_lrs_{}.npy".format(path, subset, seed), lrs)
            else:
                np.save("./{}/train_l2s_{}.npy".format(path, seed), train_losses)
                np.save("./{}/val_l2s_{}.npy".format(path, seed), val_losses)
                np.save("./{}/lrs_{}.npy".format(path, seed), lrs)
            print(f"Epoch {epoch+1}: loss = {train_loss:.6f}\t val loss = {val_loss:.6f}")

        if(epoch%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed, subset=subset)

    progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed, subset=subset)
    #val_plots(epoch, val_loader, all_val_preds, seed=seed)

    test_vals = []
    eval_loss_fn = LpLoss(2,2)
    test_value, last_metric = evaluate(test_loader, transformer, eval_loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))

    transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value, best_metric = evaluate(test_loader, transformer, eval_loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    if(subset != 'heat,adv,burger'):
        np.save("./{}/{}_test_vals_{}.npy".format(path, subset, seed), test_vals)
        np.save("./{}/metrics/{}_last_metrics_{}.npy".format(path, subset, seed), last_metric)
        np.save("./{}/metrics/{}_best_metrics_{}.npy".format(path, subset, seed), best_metric)
    else:
        np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)
        np.save("./{}/metrics/last_metric_{}.npy".format(path, seed), last_metric)
        np.save("./{}/metrics/best_metric_{}.npy".format(path, seed), best_metric)

    if(config['train_style'] == 'arbitrary_step'):
        as_rollout(test_loader, transformer, loss_fn, config, prefix, subset)
    elif(config['train_style'] == 'next_step'):
        print("\nPREFIX: {}\n".format(prefix))
        ar_rollout(test_loader, transformer, loss_fn, config, prefix, subset)

    return model_path


if __name__ == '__main__':
    # Create a transformer with an input dimension of 10, a hidden dimension
    # of 20, 2 transformer layers, and 8 attention heads.

    # Load config
    #raise
    #with open("./configs/2d_vit_config.yaml", 'r') as stream:
    #with open("./configs/lucidrains_2d_vit_config.yaml", 'r') as stream:
    #    config = yaml.safe_load(stream)
    if(sys.argv[1] == 'transolver'):
        model_name = 'transolver'
        config_name = "transolver_2d_config.yaml"
    elif(sys.argv[1] == 'vit'):
        model_name = 'vit'
        config_name = "lucidrains_2d_vit_config.yaml"
    else:
        print("Using ViT by default.")
        model_name = 'vit'
        config_path = "lucidrains_2d_vit_config.yaml"

    with open("./configs/{}".format(config_name), 'r') as stream:
        config = yaml.safe_load(stream)


    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    prefix = "2D_{}_".format(model_name) + train_args['train_style'] + "_" + train_args['dataset'] + "_" + \
             train_args['pretraining_loss'] + "_" + train_args['llm']
    prefix += "_bcs" if(train_args['bcs']) else ""
    prefix += "_coeff" if(train_args['coeff']) else ""
    prefix += "_transfer" if(train_args['transfer']) else ""
    prefix += "_sentence" if(train_args['sentence']) else ""
    prefix += "_qualitative" if(train_args['qualitative']) else ""
    train_args['prefix'] = prefix

    if(train_args['dataset'] == 'all'):
        train_args['sim_time'] = 21

    # Loop over number of samples TODO: ns = -1 is not supported in autoregressive rollout
    #for ns in [50, 100, 500, 1000]:
    #for ns in [10]:
    for ns in [50, 100]:
    #for ns in [20, 50, 100, 500, 1000]:

    #for ns in [10, 20, 50, 100, 500, 1000]:
    #for ns in [50, 100, 500, 1000]:
    #for ns in [100, 500, 1000]:
    #for ns in [10, 20, 50, 100]:
    #for ns in [500, 1000]:
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
        shutil.copy("./configs/{}_2d_config.yaml".format(model_name),
                    "{}{}_{}/{}/{}_2d_config.yaml".format(train_args['results_dir'], train_args['num_samples'],
                                                               train_args['pretraining_num_samples'], prefix, model_name))
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

            # Will run pretraining if num_pretraining_samples > 0
            model, pretrained_model_path = run_pretraining(train_args, prefix)

            torch.manual_seed(seed)
            np.random.seed(seed)
            model = get_transformer(model_name, train_args)
            if(pretrained_model_path is not None):
                model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])

            # Train on combined dataset
            train_args['dataset'] = 'all'
            transfer_model_path = run_training(model, train_args, prefix, seed, subset=train_args['dataset'])
            #if(train_args['transfer'] and train_args['dataset'] == 'all'):
            #    transfer_model_path = model_path

            # Try zero-shot and transfer learning...
            # Create the transformer model.
            #transformer = get_transformer('vit', config)
            #transformer = get_transformer('vit', train_args)

            #TODO: adjust this for the PDEBench data sets

            #if(train_args['transfer']):
            for subset in ['shallow_water', 'diffusion_reaction', 'cfd_rand_0.1_0.01_0.01', 'cfd_rand_0.1_0.1_0.1',
                           'cfd_rand_0.1_1e-8_1e-8', 'cfd_rand_1.0_0.01_0.01', 'cfd_rand_1.0_0.1_0.1', 'cfd_rand_1.0_1e-8_1e-8',
                           'cfd_turb_0.1_1e-8_1e-8', 'cfd_turb_1.0_1e-8_1e-8']:

                train_args['num_data_samples'] = 10*ns

                print("\nDATA: {}\n".format(subset))
                train_args['dataset'] = subset

                ###
                #  Either load from transfer learning path (standard training on combined data set) or from pretrained path
                #  TODO: Maybe explore pretraining THEN transfer learning?
                ###
                if(train_args['transfer']):
                    print("\nTRANSFER LEARNING FROM: {}\n".format(transfer_model_path))
                    model.load_state_dict(torch.load(transfer_model_path)['model_state_dict'])
                else:
                    print("\nFINE TUNING FROM: {}\n".format(pretrained_model_path))
                    model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])

                print("\nDOING ZERO-SHOT EVALUATION\n")
                zero_shot_evaluate(model, train_args, seed, prefix, subset=train_args['dataset'])

                print("\nFINE TUNING ON INDIVIDUAL DATA SET\n")
                model_path = run_training(model, train_args, prefix, seed, subset=train_args['dataset'])

