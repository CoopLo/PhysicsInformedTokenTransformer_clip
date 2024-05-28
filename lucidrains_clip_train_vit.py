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
from models.pitt import CLIPTransformer2D
from models.vit import CLIPVisionTransformer
from models.lucidrains_vit import CLIPViT

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


def new_get_data(config, pretraining=False, subset='heat,adv,burger'):
    train_data = PDEDataset2D(
            #path="/home/cooperlorsung/2d_heat_adv_burgers_train_large.h5",
            path="/home/cooperlorsung/NEW_HeatAdvBurgers_9216_downsampled.h5",
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
            llm=config['llm'],
            downsample=config['downsample'],
            subset=subset,
            debug=DEBUG,
    )
    val_data = PDEDataset2D(
            #path="/home/cooperlorsung/2d_heat_adv_burgers_valid_large.h5",
            #path="/home/cooperlorsung/2d_heat_adv_burgers_valid_large.h5",
            path="/home/cooperlorsung/NEW_HeatAdvBurgers_3072_downsampled.h5",
            pde="Heat, Burgers, Advection",
            mode="train",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=True,
            device='cuda:0',
            num_samples=768,
            clip=config['clip'],
            llm=config['llm'],
            downsample=config['downsample'],
            subset=subset,
            debug=DEBUG,
    )
    test_data = PDEDataset2D(
            #path="/home/cooperlorsung/2d_heat_adv_burgers_test_large.h5",
            #path="/home/cooperlorsung/2d_heat_adv_burgers_test_large.h5",
            path="/home/cooperlorsung/NEW_HeatAdvBurgers_768_downsampled.h5",
            pde="Heat, Burgers, Advection",
            mode="valid",
            resolution=[50,64,64],
            augmentation=[],
            augmentation_ratio=0.0,
            shift='None',
            load_all=True,
            device='cuda:0',
            num_samples=1000,
            clip=config['clip'],
            llm=config['llm'],
            downsample=config['downsample'],
            subset=subset,
            debug=DEBUG,
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
    #if(config['embedding'] == "standard"):
    #    print("\n USING STANDARD EMBEDDING")
    #    neural_operator = get_neural_operator(config['neural_operator'], config)
    #    transformer = StandardPhysicsInformedTokenTransformer2D(100, config['hidden'], config['layers'], config['heads'],
    #                                    output_dim1=config['num_x'], output_dim2=config['num_y'], dropout=config['dropout'],
    #                                    neural_operator=neural_operator).to(device=device)
    #elif(config['embedding'] == "novel"):
    #    print("\nUSING NOVEL EMBEDDING")
    #    neural_operator = get_neural_operator(config['neural_operator'], config)
    #    transformer = PhysicsInformedTokenTransformer2D(100, config['hidden'], config['layers'], config['heads'],
    #                                    output_dim1=config['num_x'], output_dim2=config['num_y'], dropout=config['dropout'],
    #                                    neural_operator=neural_operator).to(device=device)
    #elif(config['embedding'] == "clip"):
    #    print("\nUSING CLIP EMBEDDING")
    #    neural_operator = get_neural_operator(config['neural_operator'], config)
    #    temporal_neural_operator = get_neural_operator(config['neural_operator'], config, temporal=True)
    #    #transformer = CLIPPhysicsInformedTokenTransformer2D(100, config['hidden'], config['layers'], config['heads'],
    #    transformer = CLIPTransformer2D(100, config['hidden'], config['layers'], config['heads'],
    #                                    output_dim1=config['num_x'], output_dim2=config['num_y'], dropout=config['dropout'],
    #                                    neural_operator=neural_operator, temporal_neural_operator=temporal_neural_operator,
    #                                    latent_dim=config['latent_dim']).to(device=device)
    if(model_name == 'vit'):
        print("USING CLIP VISION TRANSFORMER\n")
        #transformer = CLIPVisionTransformer(
        #           img_size=config['img_size'],
        #           patch_size=config['patch_size'],
        #           in_chans=config['initial_step']+7 if(config['coeff']) else config['initial_step']+2,
        #           out_chans=1,
        #           embed_dim=config['embed_dim'],
        #           depth=config['depth'],
        #           n_heads=config['n_heads'],
        #           mlp_ratio=config['mlp_ratio'],
        #           qkv_bias=config['qkv_bias'],
        #           drop_rate=config['drop_rate'],
        #           attn_drop_rate=config['attn_drop_rate'],
        #           stride=config['patch_stride'],
        #           llm=config['llm'],
        #).to(device)
        transformer = CLIPViT(
                   image_size=config['img_size'],
                   patch_size=config['patch_size'],
                   #num_classes=1,
                   dim=config['dim'],
                   depth=config['depth'],
                   heads=config['heads'],
                   mlp_dim=config['mlp_dim'],
                   pool=config['pool'],
                   channels=config['initial_step'] + 2 + 5,
                   dim_head=config['dim_head'],
                   dropout=config['dropout'],
                   emb_dropout=config['emb_dropout'],
                   llm=config['llm'],
        ).to(device)


    else:
        raise ValueError("Invalid model choice.")
    return transformer


def generate_pretraining_labels(config, coeffs, y_pred):
    if(config['pretraining_loss'] == 'clip'):
        # Only interested in diagonal
        labels = torch.arange(y_pred.shape[0]).to(device)

    elif(config['pretraining_loss'] == 'weightedclip'):
        # Take magnitude-aware cosine similarity between coefficients
        nu = coeffs['nu'].unsqueeze(-1)
        ax = coeffs['ax'].unsqueeze(-1)
        ay = coeffs['ay'].unsqueeze(-1)
        cx = coeffs['cx'].unsqueeze(-1)
        cy = coeffs['cy'].unsqueeze(-1)
        coeffs = torch.cat((nu,ax,ay,cx,cy), dim=-1)
        sim_mat = torch.sqrt(torch.sum((coeffs.unsqueeze(0) * coeffs.unsqueeze(1)).abs(), dim=-1))
        norm_vec = torch.max(torch.cat((coeffs.norm(dim=-1).unsqueeze(-1),
                                        coeffs.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(coeffs.shape[0]).unsqueeze(0).to(norm_vec.device) * norm_vec.unsqueeze(1)
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(coeffs.shape[0]).unsqueeze(1).to(norm_vec.device)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]
        sim_mat /= norm_mat
        #if(DEBUG):
        #    print(coeffs)
        #    print(sim_mat)
        labels = sim_mat.clone()

    return labels


def get_pretraining_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn):

    # Select data for input and target
    if(config['train_style'] == 'next_step'):
        raise NotImplementedError("Need to implement next step.")
    elif(config['train_style'] == 'fixed_future'):
        y = x0[:, config['sim_time']]
        x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)

    # Put data on correct device
    x0 = x0.to(device).float()
    y = y.to(device).float()
    grid = grid.to(device).float()
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
    #print()
    #print()
    #print(y_pred.shape, labels.shape)
    #print(labels)
    #print(loss_fn)
    #print()
    #print()
    #raise
    loss = loss_fn(y_pred, labels)
    return loss


def save_embeddings(config, path, transformer, loader, train=True, seed=0):
    embs = []
    all_coeffs = []
    all_sim_mats = []
    with torch.no_grad():
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(loader):
            x0 = x0[:, :config['initial_step']].permute(0, 2, 3, 1)

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
    train_loader, val_loader, test_loader = new_get_data(config, pretraining=True)

    ################################################################
    # training and evaluation
    ################################################################

    _data, _, _, _ = next(iter(val_loader))
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
            loss = get_pretraining_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn)

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
                loss = get_pretraining_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn)
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
        save_embeddings(config, path, transformer, train_loader, seed=seed, train=True)
        save_embeddings(config, path, transformer, val_loader, seed=seed, train=False)

    return transformer, pretrained_model_path


def get_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn):

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
    #y_pred = transformer(torch.cat((x0, grid), dim=-1).permute(0,3,1,2), sentence_embeddings, True)
    y_pred = transformer(inp, sentence_embeddings)

    y = y.to(device=device)#.cuda()

    # Compute the loss.
    loss = loss_fn(y_pred, y)

    return y_pred, y, loss


def run_training(transformer, config, prefix, subset='heat,adv,burger'):
    path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    model_name = 'vit' + "_{}.pt".format(seed)
    if(subset != 'heat,adv,burger'):
        model_name = subset + "_" + model_name
    model_path = path + "/" + model_name

    # Create the transformer model.

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    train_loader, val_loader, test_loader = new_get_data(config, subset=subset)

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
        for bn, (x0, grid, coeffs, sentence_embeddings) in enumerate(train_loader):
            start = time.time()
            y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn)

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
                y_pred, y, loss = get_loss(config, transformer, x0, grid, coeffs, sentence_embeddings, loss_fn)
                all_val_preds.append(y_pred.detach())
                val_loss += loss_fn(y_pred, y).item()
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
    #np.save("{}{}_{}/{}/test_vals_{}.npy".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix, seed), test_vals)
    if(subset != 'heat,adv,burger'):
        np.save("./{}/{}_test_vals_{}.npy".format(path, subset, seed), test_vals)
    else:
        np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)


if __name__ == '__main__':
    # Create a transformer with an input dimension of 10, a hidden dimension
    # of 20, 2 transformer layers, and 8 attention heads.

    # Load config
    #raise
    with open("./configs/lucidrains_2d_vit_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    prefix = "2D_vit_" + train_args['train_style'] + "_" + train_args['pretraining_loss'] + \
             "_" + train_args['llm']
    prefix += "_coeff" if(train_args['coeff']) else ""
    train_args['prefix'] = prefix

    # Creat save directory
    os.makedirs("{}{}_{}/{}".format(train_args['results_dir'], train_args['num_samples'],
                                        train_args['pretraining_num_samples'], prefix), exist_ok=True)

    # Copy files to save directory
    shutil.copy("./configs/lucidrains_2d_vit_config.yaml",
                "{}{}_{}/{}/lucidrains_2d_vit_config.yaml".format(train_args['results_dir'], train_args['num_samples'],
                                                           train_args['pretraining_num_samples'], prefix))
    shutil.copy("./plot_progress.py", "{}{}_{}/{}/plot_progress.py".format(train_args['results_dir'], train_args['num_samples'],
                                                                               train_args['pretraining_num_samples'], prefix))
    shutil.copy("./pretrain_plot_progress.py", "{}{}_{}/{}/pretrain_plot_progress.py".format(train_args['results_dir'],
                             train_args['num_samples'], train_args['pretraining_num_samples'], prefix))
    shutil.copy("./finetune_plot_progress.py", "{}{}_{}/{}/finetune_plot_progress.py".format(train_args['results_dir'],
                             train_args['num_samples'], train_args['pretraining_num_samples'], prefix))


    for seed in range(train_args.pop('num_seeds')):
        print("\nSEED: {}\n".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_args['seed'] = seed
        model, pretrained_model_path = run_pretraining(train_args, prefix)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = get_transformer('vit', train_args)
        if(pretrained_model_path is not None):
            model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
        run_training(model, train_args, prefix)

        for subset in ['heat', 'burger', 'adv']:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = get_transformer('vit', train_args)
            if(pretrained_model_path is not None):
                model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
            run_training(model, train_args, prefix, subset=subset)
    
