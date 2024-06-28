import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

NUM_RUNS = 1

def load(ax, num_samples, dset):
    chans = [True, False, False, False] if(dset == 'shallow_water') else \
            [True, True, False, False] if(dset == 'diffusion_reaction') else \
            [True, True, True, True]
    chans = torch.Tensor(chans).bool()

    ###
    # No coefficient information
    ###
    # Baseline ViT
    all_trues = []
    all_preds = []
    for ns in num_samples:
        trues = []
        preds = []
        for i in range(NUM_RUNS):
            try:
                trues.append(torch.load("./{}/2D_vit_next_step_all_transfer/rollouts/{}_{}_y_trues".format(ns, i, dset)).unsqueeze(0))
                preds.append(torch.load("./{}/2D_vit_next_step_all_transfer/rollouts/{}_{}_y_preds".format(ns, i, dset)).unsqueeze(0))

                trues[-1] = trues[-1][...,chans]
                preds[-1] = preds[-1][...,chans]
            except (RuntimeError, FileNotFoundError):
                pass

        try:
            all_trues.append(torch.cat(trues, dim=0))
            all_preds.append(torch.cat(preds, dim=0))
        except RuntimeError:
            pass


    try:
        errors = [((t - p)**2) for t, p in zip(all_trues, all_preds)]
        means = torch.cat([e.mean(dim=(1,2,3,5)) for e in errors], dim=0)
        stds = torch.cat([e.std(dim=(1,2,3,5)) for e in errors], dim=0)

        xs = list(i for i in range(means.shape[1]))
        ax[0].plot(xs, means[0], lw=3, color='C0', label="ViT")
        ax[0].fill_between(xs, means[0]+stds[0], means[0]-stds[0], color='C0', alpha=0.3)
    except RuntimeError:
        pass

    # ViT + LLM
    all_trues = []
    all_preds = []
    for ns in num_samples:
        trues = []
        preds = []
        for i in range(NUM_RUNS):
            try:
                trues.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_transfer/rollouts/{}_{}_y_trues".format(ns, i, dset)).unsqueeze(0))
                preds.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_transfer/rollouts/{}_{}_y_preds".format(ns, i, dset)).unsqueeze(0))
                trues[-1] = trues[-1][...,chans]
                preds[-1] = preds[-1][...,chans]
            except (RuntimeError, FileNotFoundError):
                pass

        try:
            all_trues.append(torch.cat(trues, dim=0))
            all_preds.append(torch.cat(preds, dim=0))
        except RuntimeError:
            pass


    try:
        errors = [((t - p)**2) for t, p in zip(all_trues, all_preds)]
        means = torch.cat([e.mean(dim=(1,2,3,5)) for e in errors], dim=0)
        stds = torch.cat([e.std(dim=(1,2,3,5)) for e in errors], dim=0)

        xs = list(i for i in range(means.shape[1]))
        ax[0].plot(xs, means[0], lw=3, color='C1', label="ViT + LLM")
        ax[0].fill_between(xs, means[0]+stds[0], means[0]-stds[0], color='C1', alpha=0.3)
    except RuntimeError:
        pass

    # ViT + LLM + BCs
    all_trues = []
    all_preds = []
    for ns in num_samples:
        trues = []
        preds = []
        for i in range(NUM_RUNS):
            try:
                trues.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_transfer/rollouts/{}_{}_y_trues".format(ns, i, dset)).unsqueeze(0))
                preds.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_transfer/rollouts/{}_{}_y_preds".format(ns, i, dset)).unsqueeze(0))
                trues[-1] = trues[-1][...,chans]
                preds[-1] = preds[-1][...,chans]
            except (RuntimeError, FileNotFoundError):
                pass

        try:
            all_trues.append(torch.cat(trues, dim=0))
            all_preds.append(torch.cat(preds, dim=0))
        except RuntimeError:
            pass


    try:
        # TODO: Channel selection
        errors = [((t - p)**2) for t, p in zip(all_trues, all_preds)]
        means = torch.cat([e.mean(dim=(1,2,3,5)) for e in errors], dim=0)
        stds = torch.cat([e.std(dim=(1,2,3,5)) for e in errors], dim=0)

        xs = list(i for i in range(means.shape[1]))
        ax[0].plot(xs, means[0], lw=3, color='C2', label="ViT + LLM + BCs")
        ax[0].fill_between(xs, means[0]+stds[0], means[0]-stds[0], color='C2', alpha=0.3)
    except RuntimeError:
        pass

    ###
    # Coefficient information
    ###
    # Baseline ViT + Coeff
    all_trues = []
    all_preds = []
    for ns in num_samples:
        trues = []
        preds = []
        for i in range(NUM_RUNS):
            try:
                trues.append(torch.load("./{}/2D_vit_next_step_all_transfer_coeff/rollouts/{}_{}_y_trues".format(ns, i, dset)).unsqueeze(0))
                preds.append(torch.load("./{}/2D_vit_next_step_all_transfer_coeff/rollouts/{}_{}_y_preds".format(ns, i, dset)).unsqueeze(0))
                trues[-1] = trues[-1][...,chans]
                preds[-1] = preds[-1][...,chans]
            except (RuntimeError, FileNotFoundError):
                pass

        try:
            all_trues.append(torch.cat(trues, dim=0))
            all_preds.append(torch.cat(preds, dim=0))
        except RuntimeError:
            pass


    try:
        # TODO: Channel selection
        errors = [((t - p)**2) for t, p in zip(all_trues, all_preds)]
        means = torch.cat([e.mean(dim=(1,2,3,5)) for e in errors], dim=0)
        stds = torch.cat([e.std(dim=(1,2,3,5)) for e in errors], dim=0)
    
        xs = list(i for i in range(means.shape[1]))
        ax[1].plot(xs, means[0], lw=3, color='C0', label="ViT")
        ax[1].fill_between(xs, means[0]+stds[0], means[0]-stds[0], color='C0', alpha=0.3)
    except RuntimeError:
        pass
    
    # ViT + LLM + Coeff
    all_trues = []
    all_preds = []
    for ns in num_samples:
        trues = []
        preds = []
        for i in range(NUM_RUNS):
            try:
                trues.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_coeff_transfer/rollouts/{}_{}_y_trues".format(ns, i, dset)).unsqueeze(0))
                preds.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_coeff_transfer/rollouts/{}_{}_y_preds".format(ns, i, dset)).unsqueeze(0))
                trues[-1] = trues[-1][...,chans]
                preds[-1] = preds[-1][...,chans]
            except (RuntimeError, FileNotFoundError):
                pass

        try:
            all_trues.append(torch.cat(trues, dim=0))
            all_preds.append(torch.cat(preds, dim=0))
        except RuntimeError:
            pass

    try:
        errors = [((t - p)**2) for t, p in zip(all_trues, all_preds)]
        means = torch.cat([e.mean(dim=(1,2,3,5)) for e in errors], dim=0)
        stds = torch.cat([e.std(dim=(1,2,3,5)) for e in errors], dim=0)

        xs = list(i for i in range(means.shape[1]))
        ax[1].plot(xs, means[0], lw=3, color='C1', label="ViT + LLM")
        ax[1].fill_between(xs, means[0]+stds[0], means[0]-stds[0], color='C1', alpha=0.3)
    except RuntimeError:
        pass

    # ViT + LLM + BCs + Coeff
    all_trues = []
    all_preds = []
    for ns in num_samples:
        trues = []
        preds = []
        for i in range(NUM_RUNS):
            try:
                trues.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_coeff_transfer/rollouts/{}_{}_y_trues".format(ns, i, dset)).unsqueeze(0))
                preds.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_coeff_transfer/rollouts/{}_{}_y_preds".format(ns, i, dset)).unsqueeze(0))
                trues[-1] = trues[-1][...,chans]
                preds[-1] = preds[-1][...,chans]
            except (RuntimeError, FileNotFoundError):
                pass

        try:
            all_trues.append(torch.cat(trues, dim=0))
            all_preds.append(torch.cat(preds, dim=0))
        except RuntimeError:
            pass


    try:
        errors = [((t - p)**2) for t, p in zip(all_trues, all_preds)]
        means = torch.cat([e.mean(dim=(1,2,3,5)) for e in errors], dim=0)
        stds = torch.cat([e.std(dim=(1,2,3,5)) for e in errors], dim=0)

        xs = list(i for i in range(means.shape[1]))
        ax[1].plot(xs, means[0], lw=3, color='C2', label="ViT + LLM + BCs")
        ax[1].fill_between(xs, means[0]+stds[0], means[0]-stds[0], color='C2', alpha=0.3)
    except RuntimeError:
        pass

    # ViT + LLM + BCs + Coeff + Qual
    all_trues = []
    all_preds = []
    for ns in num_samples:
        trues = []
        preds = []
        for i in range(NUM_RUNS):
            try:
                trues.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_coeff_transfer_qualitative/rollouts/{}_{}_y_trues".format(ns, i, dset)).unsqueeze(0))
                preds.append(torch.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_coeff_transfer_qualitative/rollouts/{}_{}_y_preds".format(ns, i, dset)).unsqueeze(0))
                trues[-1] = trues[-1][...,chans]
                preds[-1] = preds[-1][...,chans]
            except FileNotFoundError:
                pass

        try:
            all_trues.append(torch.cat(trues, dim=0))
            all_preds.append(torch.cat(preds, dim=0))
        except RuntimeError:
            pass


    try:
        errors = [((t - p)**2) for t, p in zip(all_trues, all_preds)]
        means = torch.cat([e.mean(dim=(1,2,3,5)) for e in errors], dim=0)
        stds = torch.cat([e.std(dim=(1,2,3,5)) for e in errors], dim=0)

        xs = list(i for i in range(means.shape[1]))
        ax[1].plot(xs, means[0], lw=3, color='C3', label="ViT + LLM + BCs + Qual")
        ax[1].fill_between(xs, means[0]+stds[0], means[0]-stds[0], color='C3', alpha=0.3)
    except RuntimeError:
        pass


if __name__ == '__main__':

    num_samples = [10, 20, 50]
    dset = 'all'
    dsets = ['all', 'shallow_water', 'diffusion_reaction',
            'cfd_rand_0.1_0.01_0.01',
            'cfd_rand_0.1_0.1_0.1',
            'cfd_rand_0.1_1e-8_1e-8',
            'cfd_rand_1.0_0.01_0.01',
            'cfd_rand_1.0_0.1_0.1',
            'cfd_rand_1.0_1e-8_1e-8',
            'cfd_turb_0.1_1e-8_1e-8',
            'cfd_turb_1.0_1e-8_1e-8'
    ]

    for ns in num_samples:
        print("{} SAMPLES".format(ns))
        for dset in tqdm(dsets):
            fig, ax = plt.subplots(ncols=2, figsize=(15,8))
            load(ax, [ns], dset)
            ax[0].legend(loc='best')
            ax[1].legend(loc='best')

            ax[0].set_xlabel("Samples per Equation", fontsize=14)
            ax[1].set_xlabel("Samples per Equation", fontsize=14)
            ax[0].set_ylabel("Relative L2 Error", fontsize=14)

            ax[0].set_title("PDEBench {} Dataset Comparison".format(dset), fontsize=12)
            ax[1].set_title("PDEBench {} Dataset + Coeff Comparison".format(dset), fontsize=12)

            #ax[0].set_xticks(num_samples)
            #ax[1].set_xticks(num_samples)
            #ax[1].set_yscale("log")

            plt.savefig("./rollout_comparisons/{}_{}_trend.png".format(dset, ns))
            plt.close()
