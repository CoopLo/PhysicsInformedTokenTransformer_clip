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
from loss_funcs import LpLoss

from models.pitt import StandardPhysicsInformedTokenTransformer2D
from models.pitt import PhysicsInformedTokenTransformer2D
from models.pitt import CLIPPhysicsInformedTokenTransformer2D

from models.vit import VisionTransformer
#from models.vit import ViT

from models.oformer import OFormer2D, SpatialTemporalEncoder2D, STDecoder2D, PointWiseDecoder2D
from models.deeponet import DeepONet2D
from models.fno import FNO2d

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


def evaluate(test_loader, transformer, loss_fn, config=None):
    #src_mask = generate_square_subsequent_mask(640).cuda()
    with torch.no_grad():
        transformer.eval()
        test_loss = 0
        for bn, (x0, grid, coeffs) in enumerate(test_loader):
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
            y_pred = transformer(inp)
            y = y.to(device=device)
    
            # Compute the loss.
            test_loss += loss_fn(y_pred, y).item()

    return test_loss/(bn+1)


def new_get_data(config, pretraining=False, subset='heat,adv,burger'):
    train_data = PDEDataset2D(
            path="/home/cooperlorsung/NEW_HeatAdvBurgers_9216_downsampled.h5", # Used for both pretraining and finetuning
            #path="/home/cooperlorsung/2d_heat_adv_burgers_train_large.h5", 
            pde="Heat, Burgers, Advection",
            mode="train",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=False,
            device='cuda:0',
            num_samples=config['pretraining_num_samples'] if(pretraining) else config['num_samples'],
            clip=False,
            llm=None,
            downsample=config['downsample'],
            subset=subset,
            debug=DEBUG,
    )
    val_data = PDEDataset2D(
            path="/home/cooperlorsung/NEW_HeatAdvBurgers_3072_downsampled.h5",
            #path="/home/cooperlorsung/2d_heat_adv_burgers_valid_large.h5",
            pde="Heat, Burgers, Advection",
            mode="train",
            #mode="valid",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=True,
            device='cuda:0',
            num_samples=768,
            clip=False,
            llm=None,
            downsample=config['downsample'],
            subset=subset,
            debug=DEBUG,
    )
    print("\nVAL DATA\n")
    test_data = PDEDataset2D(
            path="/home/cooperlorsung/NEW_HeatAdvBurgers_768_downsampled.h5",
            #path="/home/cooperlorsung/2d_heat_adv_burgers_test_large.h5",
            pde="Heat, Burgers, Advection",
            mode="valid",
            #mode="test",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=True,
            device='cuda:0',
            num_samples=1000,
            clip=False,
            llm=None,
            downsample=config['downsample'],
            subset=subset,
            debug=DEBUG,
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
    # Create the transformer model.
    if(model_name == 'vit'):
        print("USING VISION TRANSFORMER\n")
        transformer = VisionTransformer(
                   img_size=config['img_size'],
                   patch_size=config['patch_size'],
                   in_chans=config['initial_step']+7 if(config['coeff']) else \
                            config['initial_step']+2,
                   out_chans=1,
                   embed_dim=config['embed_dim'],
                   depth=config['depth'],
                   n_heads=config['n_heads'],
                   mlp_ratio=config['mlp_ratio'],
                   qkv_bias=config['qkv_bias'],
                   drop_rate=config['drop_rate'],
                   attn_drop_rate=config['attn_drop_rate'],
                   stride=config['patch_stride'],
        ).to(device)
        #transformer = ViT(
        #           image_size=config['img_size'],
        #           patch_size=config['patch_size'],
        #           num_classes=1,
        #           dim=config['dim'],
        #           depth=config['depth'],
        #           heads=config['heads'],
        #           mlp_dim=config['mlp_dim'],
        #           pool=config['pool'],
        #           channels=1,
        #           dim_head=config['dim_head'],
        #           dropout=config['dropout'],
        #           emb_dropout=config['emb_dropout'],
        #).to(device)

                   #qkv_bias=config['qkv_bias'],
                   #attn_drop_rate=config['attn_drop_rate'],
                   #stride=config['patch_stride'],
                   #in_chans=config['initial_step']+7 if(config['coeff']) else \
                   #         config['initial_step']+2,
                   #out_chans=1,

    else:
        raise ValueError("Invalid model choice.")
    return transformer


def get_loss(config, transformer, x0, grid, coeffs, loss_fn):

    # Select data for input and target
    if(config['train_style'] == 'next_step'):
        raise NotImplementedError("Need to implement next step.")
    elif(config['train_style'] == 'fixed_future'):
        y = x0[:, config['sim_time']].unsqueeze(-1)
        x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)

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
    y_pred = transformer(inp)

    y = y.to(device=device)#.cuda()

    # Compute the loss.
    loss = loss_fn(y_pred, y)

    return y_pred, y, loss


def run_training(config, prefix, subset='heat,adv,burger'):
    path = "{}{}/{}".format(config['results_dir'], config['num_samples'], prefix)
    model_name = 'vit' + "_{}.pt".format(seed)
    if(subset != 'heat,adv,burger'):
        model_name = subset + "_" + model_name
    model_path = path + "/" + model_name

    # Create the transformer model.
    transformer = get_transformer('vit', config)

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    train_loader, val_loader, test_loader = new_get_data(config, subset=subset)

    ################################################################
    # training and evaluation
    ################################################################

    _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)


    # Use Adam as the optimizer.
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
                                                    steps_per_epoch=len(train_loader), epochs=config['epochs'])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])

    # Use mean squared error as the loss function.
    loss_fn = nn.L1Loss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = LpLoss(2,2)

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
        max_val = 0
        transformer.train()
        for bn, (x0, grid, coeffs) in enumerate(train_loader):
            start = time.time()
            y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, loss_fn)

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
            for bn, (x0, grid, coeffs) in enumerate(val_loader):
                # Forward pass: compute predictions by passing the input sequence
                y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, loss_fn)
                all_val_preds.append(y_pred.detach())
                #val_loss += loss_fn(y_pred, y).item()  # Was this the issue?

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
    test_value = evaluate(test_loader, transformer, eval_loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, transformer, eval_loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    if(subset != 'heat,adv,burger'):
        np.save("./{}/{}_test_vals_{}.npy".format(path, subset, seed), test_vals)
    else:
        np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)


if __name__ == '__main__':
    # Create a transformer with an input dimension of 10, a hidden dimension
    # of 20, 2 transformer layers, and 8 attention heads.

    # Load config
    #raise
    with open("./configs/2d_vit_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    prefix = "2D_vit_" + train_args['train_style']
    prefix += "_coeff" if(train_args['coeff']) else ""
    train_args['prefix'] = prefix

    # Creat save directory
    os.makedirs("{}{}/{}".format(train_args['results_dir'], train_args['num_samples'], prefix), exist_ok=True)

    # Copy files to save directory
    shutil.copy("./configs/2d_vit_config.yaml",
                "{}{}/{}/2d_vit_config.yaml".format(train_args['results_dir'], train_args['num_samples'], prefix))
    shutil.copy("./plot_progress.py", "{}{}/{}/plot_progress.py".format(train_args['results_dir'], train_args['num_samples'],
                                                                           prefix))
    shutil.copy("./pretrain_plot_progress.py", "{}{}/{}/pretrain_plot_progress.py".format(train_args['results_dir'],
                             train_args['num_samples'], prefix))
    shutil.copy("./finetune_plot_progress.py", "{}{}/{}/finetune_plot_progress.py".format(train_args['results_dir'],
                             train_args['num_samples'], prefix))


    for seed in range(train_args.pop('num_seeds')):
        print("\nSEED: {}\n".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_args['seed'] = seed

        #run_training(train_args, prefix)
        for subset in ['heat,adv,burger', 'heat', 'burger', 'adv']:
            run_training(train_args, prefix, subset=subset)
    
