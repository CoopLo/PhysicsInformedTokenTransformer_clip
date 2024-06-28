import numpy as np
from matplotlib import pyplot as plt

NUM_RUNS = 1

def load(ax, num_samples, dset):

    ###
    # No coefficient information
    ###
    # Baseline ViT
    all_errors = []
    for ns in num_samples:
        errors = []
        for i in range(NUM_RUNS):
            try:
                errors.append(np.load("./{}/2D_vit_next_step_all_transfer/{}_test_vals_{}.npy".format(ns, dset, i))[1])
            except FileNotFoundError:
                pass

        all_errors.append(errors)
    all_errors = [e for e in all_errors if(e != [])]
    all_errors = np.array(all_errors)

    try:
        ax[0].plot(num_samples[:len(all_errors)], all_errors.mean(axis=1))
        ax[0].errorbar(num_samples[:len(all_errors)], all_errors.mean(axis=1), yerr=all_errors.std(axis=1), capsize=3, marker='o', markersize=5,
                    lw=1, color='C0', label="ViT")
    except np.AxisError:
        pass

    # ViT + LLM
    all_errors = []
    for ns in num_samples:
        errors = []
        for i in range(NUM_RUNS):
            try:
                errors.append(np.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_transfer/{}_test_vals_{}.npy".format(ns, dset, i))[1])
            except FileNotFoundError:
                pass

        all_errors.append(errors)
    all_errors = [e for e in all_errors if(e != [])]
    all_errors = np.array(all_errors)

    try:
        ax[0].plot(num_samples[:len(all_errors)], all_errors.mean(axis=1))
        ax[0].errorbar(num_samples[:len(all_errors)], all_errors.mean(axis=1), yerr=all_errors.std(axis=1), capsize=3, marker='o', markersize=5,
                    lw=1, color='C1', label="ViT + LLM")
    except np.AxisError:
        pass

    # ViT + LLM + BCs
    all_errors = []
    for ns in num_samples:
        errors = []
        for i in range(NUM_RUNS):
            try:
                errors.append(np.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_transfer/{}_test_vals_{}.npy".format(ns, dset, i))[1])
            except FileNotFoundError:
                pass

        all_errors.append(errors)
    all_errors = [e for e in all_errors if(e != [])]
    all_errors = np.array(all_errors)

    try:
        ax[0].plot(num_samples[:len(all_errors)], all_errors.mean(axis=1))
        ax[0].errorbar(num_samples[:len(all_errors)], all_errors.mean(axis=1), yerr=all_errors.std(axis=1), capsize=3, marker='o', markersize=5,
                    lw=1, color='C2', label="ViT + LLM + BCs")
    except np.AxisError:
        pass


    ###
    # Coefficient information
    ###
    # Baseline ViT + Coeff
    all_errors = []
    for ns in num_samples:
        errors = []
        for i in range(NUM_RUNS):
            try:
                errors.append(np.load("./{}/2D_vit_next_step_all_transfer_coeff/{}_test_vals_{}.npy".format(ns, dset, i))[1])
            except FileNotFoundError:
                pass

        all_errors.append(errors)
    all_errors = [e for e in all_errors if(e != [])]
    all_errors = np.array(all_errors)

    try:
        ax[1].plot(num_samples[:len(all_errors)], all_errors.mean(axis=1))
        ax[1].errorbar(num_samples[:len(all_errors)], all_errors.mean(axis=1), yerr=all_errors.std(axis=1), capsize=3, marker='o', markersize=5,
                    lw=1, color='C0', label="ViT")
    except np.AxisError:
        pass

    # ViT + LLM + Coeff
    all_errors = []
    for ns in num_samples:
        errors = []
        for i in range(NUM_RUNS):
            try:
                errors.append(np.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_coeff_transfer/{}_test_vals_{}.npy".format(ns, dset, i))[1])
            except FileNotFoundError:
                pass

        all_errors.append(errors)
    all_errors = [e for e in all_errors if(e != [])]
    all_errors = np.array(all_errors)

    try:
        ax[1].plot(num_samples[:len(all_errors)], all_errors.mean(axis=1))
        ax[1].errorbar(num_samples[:len(all_errors)], all_errors.mean(axis=1), yerr=all_errors.std(axis=1), capsize=3, marker='o',
                       markersize=5, lw=1, color='C1', label="ViT + LLM + BCs")
    except np.AxisError:
        pass

    # ViT + LLM + BCs + Coeff
    all_errors = []
    for ns in num_samples:
        errors = []
        for i in range(NUM_RUNS):
            try:
                errors.append(np.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_coeff_transfer/{}_test_vals_{}.npy".format(ns, dset, i))[1])
            except FileNotFoundError:
                pass

        all_errors.append(errors)
    all_errors = [e for e in all_errors if(e != [])]
    all_errors = np.array(all_errors)

    try:
        ax[1].plot(num_samples[:len(all_errors)], all_errors.mean(axis=1))
        ax[1].errorbar(num_samples[:len(all_errors)], all_errors.mean(axis=1), yerr=all_errors.std(axis=1), capsize=3, marker='o', markersize=5,
                    lw=1, color='C2', label="ViT + LLM + BCs")
    except np.AxisError:
        pass

    # ViT + LLM + BCs + Coeff + Qual
    all_errors = []
    for ns in num_samples:
        errors = []
        for i in range(NUM_RUNS):
            try:
                errors.append(np.load("./{}_0/2D_clipvit_next_step_all_clip_all-mpnet-base-v2_bcs_coeff_transfer_qualitative/{}_test_vals_{}.npy".format(ns, dset, i))[1])
            except FileNotFoundError:
                pass

        all_errors.append(errors)
    all_errors = [e for e in all_errors if(e != [])]
    all_errors = np.array(all_errors)

    try:
        ax[1].plot(num_samples[:len(all_errors)], all_errors.mean(axis=1))
        ax[1].errorbar(num_samples[:len(all_errors)], all_errors.mean(axis=1), yerr=all_errors.std(axis=1), capsize=3, marker='o', markersize=5,
                    lw=1, color='C3', label="ViT + LLM + BCs + Qual")
    except np.AxisError:
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

    for dset in dsets:
        fig, ax = plt.subplots(ncols=2, figsize=(15,8))
        load(ax, num_samples, dset)
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')

        ax[0].set_xlabel("Samples per Equation", fontsize=14)
        ax[1].set_xlabel("Samples per Equation", fontsize=14)
        ax[0].set_ylabel("Relative L2 Error", fontsize=14)

        ax[0].set_title("PDEBench {} Dataset Comparison".format(dset), fontsize=12)
        ax[1].set_title("PDEBench {} Dataset + Coeff Comparison".format(dset), fontsize=12)

        ax[0].set_xticks(num_samples)
        ax[1].set_xticks(num_samples)

        plt.savefig("./{}_trend.png".format(dset))
        plt.close()
