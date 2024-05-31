import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(25,8), ncols=4)

print("Combined: ")
test_l2s = []
for i in range(5):
    try:
        ax[3].plot(np.load("./train_l2s_{}.npy".format(i)))
        ax[3].plot(np.load("./val_l2s_{}.npy".format(i)))
        train_vals = np.load("./train_l2s_{}.npy".format(i))
        val_vals = np.load("./val_l2s_{}.npy".format(i))
        test_vals = np.load("./test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

try:
    print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
except IndexError:
    print("No completed runs.")

print("\nHeat: ")
test_l2s = []
for i in range(5):
    try:
        ax[0].plot(np.load("./heat_train_l2s_{}.npy".format(i)))
        ax[0].plot(np.load("./heat_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./heat_train_l2s_{}.npy".format(i))
        val_vals = np.load("./heat_val_l2s_{}.npy".format(i))
        test_vals = np.load("./heat_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

try:
    print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
except IndexError:
    print("No completed runs.")

print("\nBurgers: ")
test_l2s = []
for i in range(5):
    try:
        ax[1].plot(np.load("./burger_train_l2s_{}.npy".format(i)))
        ax[1].plot(np.load("./burger_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./burger_train_l2s_{}.npy".format(i))
        val_vals = np.load("./burger_val_l2s_{}.npy".format(i))
        test_vals = np.load("./burger_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

try:
    print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
except IndexError:
    print("No completed runs.")

print("\nAdvection: ")
test_l2s = []
for i in range(5):
    try:
        ax[2].plot(np.load("./adv_train_l2s_{}.npy".format(i)))
        ax[2].plot(np.load("./adv_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./adv_train_l2s_{}.npy".format(i))
        val_vals = np.load("./adv_val_l2s_{}.npy".format(i))
        test_vals = np.load("./adv_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass

try:
    print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
except IndexError:
    print("No completed runs.")

ax[0].set_title("Heat", fontsize=18)
ax[1].set_title("Burgers", fontsize=18)
ax[2].set_title("Advection", fontsize=18)
ax[3].set_title("Combined", fontsize=18)
plt.show()
