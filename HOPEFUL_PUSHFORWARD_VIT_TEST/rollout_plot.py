import numpy as np
from matplotlib import pyplot as plt
import torch

fig, ax = plt.subplots(figsize=(25,16), ncols=5, nrows=2)

print("Shallow Water: ")
test_l2s = []
NUM_RUNS = 1
name = "shallow_water"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[0][0].plot(xs, mean)
        ax[0][0].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_shallow_water_rollout_mse".format(i)).cpu()
        #mean = mse[...,0].mean(dim=0)
        #std = mse[...,0].std(dim=0)
        #xs = [i for i in range(len(mean))]
        #ax[0][0].plot(xs, mean)
        #ax[0][0].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")


# Standard Training
print("\nDiffusion Reaction: ")
test_l2s = []
name = "diffusion_reaction"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[0][1].plot(xs, mean)
        ax[0][1].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_diffusion_reaction_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[0][1].plot(xs, mean)
        #ax[0][1].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

print("\ncfd_rand_0.1_0.01_0.01: ")
test_l2s = []
name = "cfd_rand_0.1_0.01_0.01"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[0][2].plot(xs, mean)
        ax[0][2].fill_between(xs, mean+std, mean-std, alpha=0.3)
        
        #fig1, ax1 = plt.subplots(ncols=4, nrows=2, figsize=(15,25))
        #ax1[0][0].imshow(trues[0,...,1,0])
        #ax1[0][1].imshow(trues[0,...,1,1])
        #ax1[0][2].imshow(trues[0,...,1,2])
        #ax1[0][3].imshow(trues[0,...,1,3])

        #ax1[1][0].imshow(preds[0,...,1,0])
        #ax1[1][1].imshow(preds[0,...,1,1])
        #ax1[1][2].imshow(preds[0,...,1,2])
        #ax1[1][3].imshow(preds[0,...,1,3])
        #plt.tight_layout()
        #plt.show()

        print("ACCUMULATED ERROR: {0}".format(sum(mean)))

        #raise
        #mse = torch.load("./rollouts/{}_cfd_rand_0.1_0.01_0.01_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[0][2].plot(xs, mean)
        #ax[0][2].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #print("ACCUMULATED ERROR: {0}".format(sum(mean)))
        ###ax[0][2].plot(np.load("./cfd_rand_0.1_0.01_0.01_train_l2s_{}.npy".format(i)))
        ###ax[0][2].plot(np.load("./cfd_rand_0.1_0.01_0.01_val_l2s_{}.npy".format(i)))
        ###train_vals = np.load("./cfd_rand_0.1_0.01_0.01_train_l2s_{}.npy".format(i))
        ###val_vals = np.load("./cfd_rand_0.1_0.01_0.01_val_l2s_{}.npy".format(i))
        ###test_vals = np.load("./cfd_rand_0.1_0.01_0.01_test_vals_{}.npy".format(i))
        ###test_l2s.append(test_vals)
        ####print(val_vals.shape)
        ###print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        ####raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

print("\ncfd_rand_0.1_0.1_0.1: ")
test_l2s = []
name = "cfd_rand_0.1_0.1_0.1"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[0][3].plot(xs, mean)
        ax[0][3].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_cfd_rand_0.1_0.1_0.1_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[0][3].plot(xs, mean)
        #ax[0][3].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
        ###ax[0][3].plot(np.load("./cfd_rand_0.1_0.1_0.1_train_l2s_{}.npy".format(i)))
        ###ax[0][3].plot(np.load("./cfd_rand_0.1_0.1_0.1_val_l2s_{}.npy".format(i)))
        ###train_vals = np.load("./cfd_rand_0.1_0.1_0.1_train_l2s_{}.npy".format(i))
        ###val_vals = np.load("./cfd_rand_0.1_0.1_0.1_val_l2s_{}.npy".format(i))
        ###test_vals = np.load("./cfd_rand_0.1_0.1_0.1_test_vals_{}.npy".format(i))
        ###test_l2s.append(test_vals)
        ####print(val_vals.shape)
        ###print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        ####raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

print("\ncfd_rand_0.1_1e-8_1e-8: ")
test_l2s = []
name = "cfd_rand_0.1_1e-8_1e-8"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[0][4].plot(xs, mean)
        ax[0][4].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_cfd_rand_0.1_1e-8_1e-8_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[0][4].plot(xs, mean)
        #ax[0][4].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
        ###ax[0][4].plot(np.load("./cfd_rand_0.1_1e-8_1e-8_train_l2s_{}.npy".format(i)))
        ###ax[0][4].plot(np.load("./cfd_rand_0.1_1e-8_1e-8_val_l2s_{}.npy".format(i)))
        ###train_vals = np.load("./cfd_rand_0.1_1e-8_1e-8_train_l2s_{}.npy".format(i))
        ###val_vals = np.load("./cfd_rand_0.1_1e-8_1e-8_val_l2s_{}.npy".format(i))
        ###test_vals = np.load("./cfd_rand_0.1_1e-8_1e-8_test_vals_{}.npy".format(i))
        ###test_l2s.append(test_vals)
        ####print(val_vals.shape)
        ###print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        ####raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

print("\ncfd_rand_1.0_0.01_0.01: ")
test_l2s = []
name = "cfd_rand_1.0_0.01_0.01"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[1][2].plot(xs, mean)
        ax[1][2].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_cfd_rand_1.0_0.01_0.01_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[1][2].plot(xs, mean)
        #ax[1][2].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
        ###ax[1][2].plot(np.load("./cfd_rand_1.0_0.01_0.01_train_l2s_{}.npy".format(i)))
        ###ax[1][2].plot(np.load("./cfd_rand_1.0_0.01_0.01_val_l2s_{}.npy".format(i)))
        ###train_vals = np.load("./cfd_rand_1.0_0.01_0.01_train_l2s_{}.npy".format(i))
        ###val_vals = np.load("./cfd_rand_1.0_0.01_0.01_val_l2s_{}.npy".format(i))
        ###test_vals = np.load("./cfd_rand_1.0_0.01_0.01_test_vals_{}.npy".format(i))
        ###test_l2s.append(test_vals)
        ####print(val_vals.shape)
        ###print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        ####raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

print("\ncfd_rand_1.0_0.1_0.1: ")
test_l2s = []
name = "cfd_rand_1.0_0.1_0.1"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[1][3].plot(xs, mean)
        ax[1][3].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_cfd_rand_1.0_0.1_0.1_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[1][3].plot(xs, mean)
        #ax[1][3].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
        ###ax[1][3].plot(np.load("./cfd_rand_1.0_0.1_0.1_train_l2s_{}.npy".format(i)))
        ###ax[1][3].plot(np.load("./cfd_rand_1.0_0.1_0.1_val_l2s_{}.npy".format(i)))
        ###train_vals = np.load("./cfd_rand_1.0_0.1_0.1_train_l2s_{}.npy".format(i))
        ###val_vals = np.load("./cfd_rand_1.0_0.1_0.1_val_l2s_{}.npy".format(i))
        ###test_vals = np.load("./cfd_rand_1.0_0.1_0.1_test_vals_{}.npy".format(i))
        ###test_l2s.append(test_vals)
        ####print(val_vals.shape)
        ###print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        ####raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

print("\ncfd_rand_1.0_1e-8_1e-8: ")
test_l2s = []
name = "cfd_rand_1.0_1e-8_1e-8"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[1][4].plot(xs, mean)
        ax[1][4].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_cfd_rand_1.0_1e-8_1e-8_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[1][4].plot(xs, mean)
        #ax[1][4].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
        ###ax[1][4].plot(np.load("./cfd_rand_1.0_1e-8_1e-8_train_l2s_{}.npy".format(i)))
        ###ax[1][4].plot(np.load("./cfd_rand_1.0_1e-8_1e-8_val_l2s_{}.npy".format(i)))
        ###train_vals = np.load("./cfd_rand_1.0_1e-8_1e-8_train_l2s_{}.npy".format(i))
        ###val_vals = np.load("./cfd_rand_1.0_1e-8_1e-8_val_l2s_{}.npy".format(i))
        ###test_vals = np.load("./cfd_rand_1.0_1e-8_1e-8_test_vals_{}.npy".format(i))
        ###test_l2s.append(test_vals)
        ####print(val_vals.shape)
        ###print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        ####raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

print("\ncfd_turb_0.1_1e-8_1e-8: ")
test_l2s = []
name = "cfd_turb_0.1_1e-8_1e-8"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[1][0].plot(xs, mean)
        ax[1][0].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_cfd_turb_0.1_1e-8_1e-8_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[1][0].plot(xs, mean)
        #ax[1][0].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
        ###ax[1][0].plot(np.load("./cfd_turb_0.1_1e-8_1e-8_train_l2s_{}.npy".format(i)))
        ###ax[1][0].plot(np.load("./cfd_turb_0.1_1e-8_1e-8_val_l2s_{}.npy".format(i)))
        ###train_vals = np.load("./cfd_turb_0.1_1e-8_1e-8_train_l2s_{}.npy".format(i))
        ###val_vals = np.load("./cfd_turb_0.1_1e-8_1e-8_val_l2s_{}.npy".format(i))
        ###test_vals = np.load("./cfd_turb_0.1_1e-8_1e-8_test_vals_{}.npy".format(i))
        ###test_l2s.append(test_vals)
        ####print(val_vals.shape)
        ###print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        ####raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

print("\ncfd_turb_1.0_1e-8_1e-8: ")
test_l2s = []
name = "cfd_turb_1.0_1e-8_1e-8"
for i in range(NUM_RUNS):
    try:
        preds = torch.load("./rollouts/{}_{}_y_preds".format(i, name)).cpu()
        trues = torch.load("./rollouts/{}_{}_y_trues".format(i, name)).cpu()
        errors = ((trues - preds)**2).mean(dim=(1,2,4))
        mean = errors.mean(dim=0)
        std = errors.std(dim=0)
        xs = [i for i in range(len(mean))]
        ax[1][1].plot(xs, mean)
        ax[1][1].fill_between(xs, mean+std, mean-std, alpha=0.3)
        #mse = torch.load("./rollouts/{}_cfd_turb_1.0_1e-8_1e-8_rollout_mse".format(i)).cpu()
        #mean = mse[...,:2].mean(dim=(0,2))
        #std = mse[...,:2].std(dim=(0,2))
        #xs = [i for i in range(len(mean))]
        #ax[1][1].plot(xs, mean)
        #ax[1][1].fill_between(xs, mean+std, mean-std, alpha=0.3)
        print("ACCUMULATED ERROR: {0}".format(sum(mean)))
        ###ax[1][1].plot(np.load("./cfd_turb_1.0_1e-8_1e-8_train_l2s_{}.npy".format(i)))
        ###ax[1][1].plot(np.load("./cfd_turb_1.0_1e-8_1e-8_val_l2s_{}.npy".format(i)))
        ###train_vals = np.load("./cfd_turb_1.0_1e-8_1e-8_train_l2s_{}.npy".format(i))
        ###val_vals = np.load("./cfd_turb_1.0_1e-8_1e-8_val_l2s_{}.npy".format(i))
        ###test_vals = np.load("./cfd_turb_1.0_1e-8_1e-8_test_vals_{}.npy".format(i))
        ###test_l2s.append(test_vals)
        ####print(val_vals.shape)
        ###print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        ####raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

if(NUM_RUNS > 1):
    try:
        print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
    except IndexError:
        print("No completed runs.")

ax[0][0].set_title("Shallow Water", fontsize=18)
ax[0][1].set_title("Diffusion Reaction", fontsize=18)
ax[0][2].set_title("cfd_rand_0.1_0.01_0.01", fontsize=18)
ax[0][3].set_title("cfd_rand_0.1_0.1_0.1", fontsize=18)
ax[0][4].set_title("cfd_rand_0.1_1e-8_1e-8", fontsize=18)
ax[1][0].set_title("cfd_turb_0.1_1e-8_1e-8", fontsize=18)
ax[1][1].set_title("cfd_turb_1.0_1e-8_1e-8", fontsize=18)
ax[1][2].set_title("cfd_rand_1.0_0.01_0.01", fontsize=18)
ax[1][3].set_title("cfd_rand_1.0_0.1_0.1", fontsize=18)
ax[1][4].set_title("cfd_rand_1.0_1e-8_1e-8", fontsize=18)
plt.tight_layout()
plt.savefig("./rollout_plots.png")
#plt.show()
