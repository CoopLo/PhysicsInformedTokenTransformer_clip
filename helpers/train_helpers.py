import torch
import numpy as np
from models.pitt import LLMPITT2D, LLMPITT2D
from models.transolver import EmbeddingTransolver
from models.dpot import DPOTNet, LLMDPOTNet
from models.lucidrains_vit import ViT, CLIPViT, LLMCLIPViT
from metrics import metric_func
from tqdm import tqdm
from pdebench_data_handling import MultiDataset


def make_prediction(config, transformer, x0, grid, t, coeffs=None, sentence_embeddings=None):
    if(config['coeff']): # Stack coefficients
        # Fix size of coefficients
        coeffs = coeffs[:,None,None,:].tile(1, x0.shape[1], x0.shape[2], 1)

        # Stack coefficients onto values
        if(len(x0.shape) == 5): # Hacked together
            x0 = x0.flatten(3,4)
        x0 = torch.cat((x0, coeffs), dim=-1)#.permute(0,3,1,2)
    else:
        if(not isinstance(transformer, (DPOTNet, LLMDPOTNet))):
            if(len(x0.shape) == 5): # Hacked together
                x0 = x0.flatten(3,4)
            #inp = torch.cat((x0, grid), dim=-1).permute(0,3,1,2)

    if(sentence_embeddings is not None):
        if(isinstance(transformer, EmbeddingTransolver)):
            y_pred = transformer(grid, x0, sentence_embeddings)
        elif(isinstance(transformer, LLMPITT2D)):
            y_pred = transformer(grid, sentence_embeddings, x0, t)
        elif(isinstance(transformer, LLMDPOTNet)):
            y_pred, cls_pred = transformer(x0, sentence_embeddings)
            y_pred = y_pred[...,0,:]
        else:
            y_pred = transformer(x0, sentence_embeddings)[...,0].permute(0,2,3,1)
    else:
        if(isinstance(transformer, DPOTNet)):
            # B, X, Y, T, C
            y_pred, cls_pred = transformer(x0)
            y_pred = y_pred[...,0,:]
            #TODO: Figure out class prediction
        elif(isinstance(transformer, ViT)):
            y_pred = transformer(x0)

    return y_pred


def get_loss(config, transformer, x0, grid, coeffs, loss_fn, sentence_embeddings=None, times=None, evaluate=False):
    device = config['device']
    #print()
    #print()
    #print(x0.shape)
    #print()
    #print()

    # Select data for input and target
    if(config['train_style'] == 'next_step'):
        steps = torch.Tensor(np.random.choice(range(config['initial_step'], config['sim_time']), x0.shape[0])).long()
        y = torch.cat([x0[idx,:,:,i][None] for idx, i in enumerate(steps)], dim=0)

        if(isinstance(transformer, (DPOTNet, LLMDPOTNet))):
            x0 = torch.cat([x0[idx,:,:,i-config['initial_step']:i][None] for idx, i in enumerate(steps)], dim=0)
        else:
            x0 = torch.cat([x0[idx,:,:,i-config['initial_step']:i][None] for idx, i in enumerate(steps)], dim=0).flatten(3,4)

        if(not isinstance(times, torch.Tensor)):
            times = torch.Tensor(times)
        if(isinstance(transformer, LLMPITT2D)):
            # TODO Fix this. Should be dt, coeffs come later
            t = torch.cat([times[idx, i][None] for idx, i in enumerate(steps)], dim=0)
        else:
            t = None

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

    y_pred = make_prediction(config, transformer, x0, grid, t, coeffs=coeffs, sentence_embeddings=sentence_embeddings)

    # Compute the loss.
    if(evaluate):
        ch0 = not((y[...,-1,0] == 0).all())
        ch1 = not((y[...,-1,1] == 0).all())
        ch2 = not((y[...,-1,2] == 0).all())
        ch3 = not((y[...,-1,3] == 0).all())
        chans = [ch0, ch1, ch2, ch3]
        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(y_pred[...,chans], y[...,chans], if_mean=True,
                                                                           Lx=1., Ly=1., Lz=1.)
        loss = loss_fn(y_pred, y)
        return y_pred, y, loss, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F
    else:
        loss = loss_fn(y_pred, y)
        return y_pred, y, loss


NOISE_SCALE = 1.e-6
def get_dpot_loss(config, ep, transformer, x0, grid, coeffs, loss_fn, sentence_embeddings=None, times=None, evaluate=False):
    '''
        Mimics training from DPOT paper. Injects noise, uses temporal bundling, also pushforward
        (looks like they only ever use a temporal bundle of 1 though...)

        NOTE: Currently only supports t_bundle = 1
    '''
    device = config['device']
    horizon = min(config['pushforward'], ep+1)

    # Select data for input and target
    steps = torch.Tensor(np.random.choice(range(config['initial_step'], config['sim_time']-horizon), x0.shape[0])).long()
    inp = torch.cat([x0[idx,:,:,i-config['initial_step']:i][None] for idx, i in enumerate(steps)], dim=0)

    #inp = x0[...,:config['initial_step'],:]

    t = times[:,0] if(isinstance(transformer, LLMPITT2D)) else None
    loss = 0
    for step in range(0, horizon, config['t_bundle']):
        #print()
        #print(step, inp.shape)
        # Inject noise
        inp = inp + NOISE_SCALE * torch.sum(inp**2, dim=(1,2,3), keepdim=True)**0.5 * torch.randn_like(inp)

        # Make prediction
        y_pred = make_prediction(config, transformer, inp, grid, t, coeffs=coeffs, sentence_embeddings=sentence_embeddings)

        # Get ground truth value
        target_step = config['initial_step'] + step
        #y = x0[...,target_step:target_step + config['t_bundle'],:]
        #print(x0.shape, steps)
        y = torch.cat([x0[idx,:,:,i+step:i+step+config['t_bundle'],:][None] for idx, i in enumerate(steps)], dim=0)

        #print(y_pred.shape, y.shape)
        sample_loss = loss_fn(y_pred, y)
        if(sample_loss.isnan()):
            break
        else:
            loss += sample_loss
        #print(loss)

        # Update input
        inp = torch.cat((inp[..., config['t_bundle']:,:], y_pred.unsqueeze(3)), dim=-2)

        # Stack predictions (seems unecessary?)
        #if(step == 0):
        #    pred = y_pred
        #else:
        #    pred = torch.cat((pred, y_pred), dim=-2)

        #raise

    # Compute the loss.
    if(evaluate):
        ch0 = not((y[...,-1,0] == 0).all())
        ch1 = not((y[...,-1,1] == 0).all())
        ch2 = not((y[...,-1,2] == 0).all())
        ch3 = not((y[...,-1,3] == 0).all())
        chans = [ch0, ch1, ch2, ch3]
        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(y_pred.unsqueeze(3)[...,chans], y[...,chans],
                                                                           if_mean=True, Lx=1., Ly=1., Lz=1.)
        return y_pred, y, loss, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F
    else:
        return y_pred, y, loss


def as_rollout(test_loader, transformer, loss_fn, config, prefix, subset):
    device = config['device']
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


def ar_rollout(test_loader, transformer, loss_fn, config, prefix, subset, seed=0):
    #TODO: Make robust to choosing coefficients
    device = config['device']
    all_y_preds, all_y_trues = [], []
    with torch.no_grad():
        transformer.eval()
        test_loss = 0

        # TODO: Loop over dataset not data loader
        print("Autoregressive rollout...")
        for idx in tqdm(range(test_loader.dataset.data.shape[0])):
            #if(not isinstance(transformer, DPOTNet)):
            if(isinstance(transformer, (DPOTNet, LLMDPOTNet))):
                x0 = test_loader.dataset.data[idx][...,:config['initial_step'],:].unsqueeze(0)
            else:
                x0 = test_loader.dataset.data[idx][...,:config['initial_step'],:].unsqueeze(0).flatten(3,4)
            #x0 = test_loader.dataset.data[idx][...,:config['initial_step'],:].unsqueeze(0)
            y = test_loader.dataset.data[idx][...,config['initial_step']:,:].unsqueeze(0)
            if(isinstance(transformer, LLMPITT2D)):
                if(test_loader.dataset.dt.shape[0] == test_loader.dataset.data.shape[0]):
                    t = test_loader.dataset.dt[idx][0].unsqueeze(0)
                else:
                    t = test_loader.dataset.dt[0][0].unsqueeze(0)
            else:
                t = None

            # Grid changes based on multi or single dataset.
            if(isinstance(test_loader.dataset, MultiDataset)):
                #TODO: This no longer supports unevenly sized data sets
                sample_idx = idx%(int(0.1*config['num_samples']))
                dset_idx = idx//int((0.1*config['num_samples']))

                try:
                    grid = test_loader.dataset.grids[dset_idx].unsqueeze(0)
                except IndexError:
                    print(test_loader.dataset.grids.shape)
                    print(test_loader.dataset.data.shape)
                    print(test_loader.dataset.dt.shape)
                    raise
                if(not isinstance(transformer, (DPOTNet, ViT))):
                    sentence_embeddings = torch.Tensor(test_loader.dataset.dsets[dset_idx].sentence_embeddings[sample_idx]).unsqueeze(0)
                else:
                    sentence_embeddings = None

                coeffs = test_loader.dataset.coeff[dset_idx].unsqueeze(0)
            else:
                grid = test_loader.dataset.grid.unsqueeze(0)
                if(not isinstance(transformer, (DPOTNet, ViT))):
                    sentence_embeddings = test_loader.dataset.sentence_embeddings[idx].unsqueeze(0)
                else:
                    sentence_embeddings = None
                coeffs = test_loader.dataset.coeffs.unsqueeze(0)

            if(len(x0.shape) == 5 and not isinstance(transformer, (DPOTNet, LLMDPOTNet))):
                x0 = x0[...,0,:]

            y_preds = []
            for i in range(y.shape[-2]):


                # Make prediction
                #coeffs = None #TODO: Need to make this option viable.

                #print()
                #print(x0.shape, grid.shape, coeffs.shape)
                y_pred = make_prediction(config, transformer, x0, grid, t, coeffs=coeffs, sentence_embeddings=sentence_embeddings)
                if(isinstance(transformer, EmbeddingTransolver)):
                    #y_pred = transformer(grid, x0, sentence_embeddings)
                    x0 = torch.cat((x0, y_pred), dim=-1)[...,-transformer.space_dim:]
                elif(isinstance(transformer, LLMPITT2D)):
                    #y_pred = transformer(grid, sentence_embeddings, x0, t)
                    x0 = torch.cat((x0, y_pred), dim=-1)[...,-transformer.neural_operator.channels:]
                elif(isinstance(transformer, (DPOTNet, LLMDPOTNet))):
                    #print(x0.shape, y_pred.shape)
                    #print(y_pred.unsqueeze(3).shape)
                    x0 = torch.cat((x0, y_pred.unsqueeze(3)), dim=-2)[...,-transformer.in_timesteps:,:]
                    #print(x0.shape)
                    #raise
                else:
                    #y_pred = transformer(inp).permute(0,4,2,3,1)[0]
                    shift = -transformer.channels + 5 if(config['coeff']) else -transformer.channels 
                    #print("SHAPE: {} + {} \tSHIFT: {}".format(x0.shape, y_pred.shape, shift))
                    x0 = torch.cat((x0, y_pred), dim=-1)[...,shift:]
                #print()

                # Save preds
                y_preds.append(y_pred.unsqueeze(0))

            # Save data and preds
            all_y_preds.append(torch.cat(y_preds, dim=1))
            all_y_trues.append(y)

    # Stack predictions and ground truth
    all_y_preds = torch.cat(all_y_preds, dim=0).permute(0,2,3,1,4)
    all_y_trues = torch.cat(all_y_trues, dim=0)

    # Now in shape traj x time x space x channels
    #mse = ((all_y_preds - all_y_trues)**2).mean(dim=(0,2))
    mse = ((all_y_preds - all_y_trues)**2).mean(dim=(1,2,4))

    # Save relevant info
    #path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'],
    #                           config['pretraining_num_samples'], prefix)
    if(not isinstance(transformer, (DPOTNet, ViT))):
        path = "{}{}_{}/{}".format(config['results_dir'], config['num_samples'], config['pretraining_num_samples'], prefix)
    else:
        path = "{}{}/{}".format(config['results_dir'], config['num_samples'], prefix)

    if(subset != 'heat,adv,burger'):
        torch.save(mse, path+"/rollouts/{}_{}_rollout_mse".format(seed, subset))
        #torch.save(all_y_trues.cpu(), path+"/rollouts/{}_{}_y_trues".format(seed, subset))
        #torch.save(all_y_preds.cpu(), path+"/rollouts/{}_{}_y_preds".format(seed, subset))
    else:
        torch.save(mse, path+"/rollouts/{}_rollout_mse".format(seed))
        #torch.save(all_y_trues.cpu(), path+"/rollouts/{}_all_y_trues".format(seed))
        #torch.save(all_y_preds.cpu(), path+"/rollouts/{}_all_y_preds".format(seed))
    return test_loss/(idx+1)


