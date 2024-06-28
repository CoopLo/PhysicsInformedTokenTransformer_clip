import numpy as np
from matplotlib import pyplot as plt
import json


def fix(metrics):
    for key, val in metrics.items():
        new_val = [v.cpu().numpy() for v in val]
        metrics[key] = np.array(new_val)
    return metrics

fig, ax = plt.subplots(figsize=(25,16), ncols=5, nrows=2)

print("Shallow Water: ")
test_l2s = []
NUM_RUNS = 1
for i in range(NUM_RUNS):
    try:
        metrics = np.load("./metrics/shallow_water_best_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))

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
        metrics = np.load("./metrics/diffusion_reaction_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))

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
        metrics = np.load("./metrics/cfd_rand_0.1_0.01_0.01_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))

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
        metrics = np.load("./metrics/cfd_rand_0.1_0.1_0.1_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))
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
        metrics = np.load("./metrics/cfd_rand_0.1_1e-8_1e-8_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))
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
        metrics = np.load("./metrics/cfd_rand_1.0_0.01_0.01_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))
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
        metrics = np.load("./metrics/cfd_rand_1.0_0.1_0.1_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))
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
        metrics = np.load("./metrics/cfd_rand_1.0_1e-8_1e-8_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))
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
        metrics = np.load("./metrics/cfd_turb_0.1_1e-8_1e-8_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))
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
        metrics = np.load("./metrics/cfd_turb_0.1_1e-8_1e-8_last_metrics_{}.npy".format(i),
                          allow_pickle=True).astype(object)
        metrics = dict(metrics.tolist())
        metrics = fix(metrics)

        for key, val in metrics.items():
            if(np.inf in val):
                print("{0}:\tINF pm INF".format(key))
            else:
                print("{0}:\t{1:.2e} pm {2:.2e}".format(key, val.mean(), val.std()))
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
#plt.savefig("./all_progress_plots.png")
#plt.show()
