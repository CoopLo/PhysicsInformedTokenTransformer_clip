import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(25,16), ncols=5, nrows=2)

print("Shallow Water: ")
test_l2s = []
NUM_RUNS = 1
for i in range(NUM_RUNS):
    try:
        ax[0][0].plot(np.load("./shallow_water_train_l2s_{}.npy".format(i)))
        ax[0][0].plot(np.load("./shallow_water_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./shallow_water_train_l2s_{}.npy".format(i))
        val_vals = np.load("./shallow_water_val_l2s_{}.npy".format(i))
        test_vals = np.load("./shallow_water_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[0][1].plot(np.load("./diffusion_reaction_train_l2s_{}.npy".format(i)))
        ax[0][1].plot(np.load("./diffusion_reaction_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./diffusion_reaction_train_l2s_{}.npy".format(i))
        val_vals = np.load("./diffusion_reaction_val_l2s_{}.npy".format(i))
        test_vals = np.load("./diffusion_reaction_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[0][2].plot(np.load("./cfd_rand_0.1_0.01_0.01_train_l2s_{}.npy".format(i)))
        ax[0][2].plot(np.load("./cfd_rand_0.1_0.01_0.01_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./cfd_rand_0.1_0.01_0.01_train_l2s_{}.npy".format(i))
        val_vals = np.load("./cfd_rand_0.1_0.01_0.01_val_l2s_{}.npy".format(i))
        test_vals = np.load("./cfd_rand_0.1_0.01_0.01_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[0][3].plot(np.load("./cfd_rand_0.1_0.1_0.1_train_l2s_{}.npy".format(i)))
        ax[0][3].plot(np.load("./cfd_rand_0.1_0.1_0.1_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./cfd_rand_0.1_0.1_0.1_train_l2s_{}.npy".format(i))
        val_vals = np.load("./cfd_rand_0.1_0.1_0.1_val_l2s_{}.npy".format(i))
        test_vals = np.load("./cfd_rand_0.1_0.1_0.1_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[0][4].plot(np.load("./cfd_rand_0.1_1e-8_1e-8_train_l2s_{}.npy".format(i)))
        ax[0][4].plot(np.load("./cfd_rand_0.1_1e-8_1e-8_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./cfd_rand_0.1_1e-8_1e-8_train_l2s_{}.npy".format(i))
        val_vals = np.load("./cfd_rand_0.1_1e-8_1e-8_val_l2s_{}.npy".format(i))
        test_vals = np.load("./cfd_rand_0.1_1e-8_1e-8_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[1][2].plot(np.load("./cfd_rand_1.0_0.01_0.01_train_l2s_{}.npy".format(i)))
        ax[1][2].plot(np.load("./cfd_rand_1.0_0.01_0.01_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./cfd_rand_1.0_0.01_0.01_train_l2s_{}.npy".format(i))
        val_vals = np.load("./cfd_rand_1.0_0.01_0.01_val_l2s_{}.npy".format(i))
        test_vals = np.load("./cfd_rand_1.0_0.01_0.01_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[1][3].plot(np.load("./cfd_rand_1.0_0.1_0.1_train_l2s_{}.npy".format(i)))
        ax[1][3].plot(np.load("./cfd_rand_1.0_0.1_0.1_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./cfd_rand_1.0_0.1_0.1_train_l2s_{}.npy".format(i))
        val_vals = np.load("./cfd_rand_1.0_0.1_0.1_val_l2s_{}.npy".format(i))
        test_vals = np.load("./cfd_rand_1.0_0.1_0.1_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[1][4].plot(np.load("./cfd_rand_1.0_1e-8_1e-8_train_l2s_{}.npy".format(i)))
        ax[1][4].plot(np.load("./cfd_rand_1.0_1e-8_1e-8_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./cfd_rand_1.0_1e-8_1e-8_train_l2s_{}.npy".format(i))
        val_vals = np.load("./cfd_rand_1.0_1e-8_1e-8_val_l2s_{}.npy".format(i))
        test_vals = np.load("./cfd_rand_1.0_1e-8_1e-8_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[1][0].plot(np.load("./cfd_turb_0.1_1e-8_1e-8_train_l2s_{}.npy".format(i)))
        ax[1][0].plot(np.load("./cfd_turb_0.1_1e-8_1e-8_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./cfd_turb_0.1_1e-8_1e-8_train_l2s_{}.npy".format(i))
        val_vals = np.load("./cfd_turb_0.1_1e-8_1e-8_val_l2s_{}.npy".format(i))
        test_vals = np.load("./cfd_turb_0.1_1e-8_1e-8_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
for i in range(NUM_RUNS):
    try:
        ax[1][1].plot(np.load("./cfd_turb_1.0_1e-8_1e-8_train_l2s_{}.npy".format(i)))
        ax[1][1].plot(np.load("./cfd_turb_1.0_1e-8_1e-8_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./cfd_turb_1.0_1e-8_1e-8_train_l2s_{}.npy".format(i))
        val_vals = np.load("./cfd_turb_1.0_1e-8_1e-8_val_l2s_{}.npy".format(i))
        test_vals = np.load("./cfd_turb_1.0_1e-8_1e-8_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
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
plt.savefig("./all_progress_plots.png")
plt.show()
