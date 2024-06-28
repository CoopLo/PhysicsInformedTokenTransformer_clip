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

from models.pitt import StandardPhysicsInformedTokenTransformer2D, LLMPITT2D
from models.pitt import PhysicsInformedTokenTransformer2D
from models.pitt import CLIPPhysicsInformedTokenTransformer2D

#from models.vit import VisionTransformer
from models.lucidrains_vit import ViT, CLIPViT, LLMCLIPViT

from models.oformer import OFormer2D, SpatialTemporalEncoder2D, STDecoder2D, PointWiseDecoder2D
from models.deeponet import DeepONet2D
from models.fno import FNO2d
from models.transolver import EmbeddingTransolver

from helpers import get_data, get_transformer, get_loss, get_dpot_loss, as_rollout, ar_rollout
from metrics import metric_func

import sys

device = 'cuda' if(torch.cuda.is_available()) else 'cpu'

DEBUG = True

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


def run_pretraining(config, prefix, model="vit"):
    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    pretrained_path = "{}{}/{}".format(config['pretrained_model_path'], config['pretraining_num_samples'],
                                       prefix)

    model_name = 'pretraining' + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    pretrained_model_path = pretrained_path + "/" + model_name

    # Create the transformer model.
    transformer = get_transformer(model, config)
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
        for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(train_loader):
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
            for bn, (x0, grid, coeffs, dt, sentence_embeddings) in enumerate(val_loader):
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


def run_training(transformer, config, prefix, seed, subset='heat,adv,burger'):
    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    model_name = 'vit' + "_{}.pt".format(seed)
    if(subset != 'heat,adv,burger'):
        model_name = subset + "_" + model_name
    model_path = path + "/" + model_name

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    train_loader, val_loader, test_loader = get_data(config)

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
            y_pred, y, loss = get_dpot_loss(config, epoch, transformer, x0, grid, coeffs, loss_fn,
                                       sentence_embeddings=sentence_embeddings,
                                       times=train_loader.dataset.dt)

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
                    y_pred, y, loss = get_dpot_loss(config, epoch, transformer, x0, grid, coeffs, loss_fn,
                                               sentence_embeddings=sentence_embeddings,
                                               times=train_loader.dataset.dt)
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
        ar_rollout(test_loader, transformer, loss_fn, config, prefix, subset, seed=seed)

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
        model_name = 'clipvit'
        config_name = "lucidrains_2d_vit_config.yaml"
    elif(sys.argv[1] == 'pitt'):
        model_name = 'pitt'
        config_name = "pitt_2d_config.yaml"
    elif(sys.argv[1] == 'dpot'):
        model_name = 'llmdpot'
        config_name = "dpot_2d_config.yaml"
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
    prefix += "_DEBUG" if(train_args['DEBUG']) else ""

    train_args['prefix'] = prefix

    if(train_args['dataset'] == 'all'):
        train_args['sim_time'] = 21

    # Loop over number of samples TODO: ns = -1 is not supported in autoregressive rollout
    #for ns in [10, 20, 50, 100]:
    #for ns in [10]:
    for ns in [10, 20, 50]:#, 100]:
    #for ns in [20, 50]:#, 100]:
    #for ns in [50]:#, 100]:
    #for ns in [100]:

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
                model, pretrained_model_path = run_pretraining(train_args, prefix, model=model_name)
                print("\n\nPRETRIANED MODEL PATH: {}\n\n".format(pretrained_model_path))

            torch.manual_seed(seed)
            np.random.seed(seed)
            model = get_transformer(model_name, train_args)
            if(pretrained_model_path is not None):
                model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])

            # Train on combined dataset
            train_args['dataset'] = 'all'
            if(not train_args['DEBUG']):
                transfer_model_path = run_training(model, train_args, prefix, seed, subset=train_args['dataset'])

            #if(train_args['transfer']):
            #for subset in ['diffusion_reaction', 'cfd_rand_0.1_0.01_0.01', 'cfd_rand_0.1_0.1_0.1',
            #for subset in ['cfd_rand_0.1_0.01_0.01', 'cfd_rand_0.1_0.1_0.1',
            for subset in ['shallow_water', 'diffusion_reaction', 'cfd_rand_0.1_0.01_0.01', 'cfd_rand_0.1_0.1_0.1',
                           'cfd_rand_0.1_1e-8_1e-8', 'cfd_rand_1.0_0.01_0.01', 'cfd_rand_1.0_0.1_0.1', 'cfd_rand_1.0_1e-8_1e-8',
                           'cfd_turb_0.1_1e-8_1e-8', 'cfd_turb_1.0_1e-8_1e-8']:

                if(train_args['DEBUG']):
                    train_args['dataset'] = 'cfd_rand_0.1_0.01_0.01'
                    train_args['num_data_samples'] = 50
                else:
                    print("\nDATA: {}\n".format(subset))
                    train_args['dataset'] = subset
                    train_args['num_data_samples'] = 20*ns

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

                if(not train_args['DEBUG']):
                    print("\nDOING ZERO-SHOT EVALUATION\n")
                    zero_shot_evaluate(model, train_args, seed, prefix, subset=train_args['dataset'])

                print("\nFINE TUNING ON INDIVIDUAL DATA SET\n")
                model_path = run_training(model, train_args, prefix, seed, subset=train_args['dataset'])

                if(train_args['DEBUG']):
                    break

