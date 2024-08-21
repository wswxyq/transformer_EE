"""
Make plots of the loss.
"""

import os
from matplotlib import pyplot as plt


def plot_loss(train_loss_list_per_epoch, valid_loss_list_per_epoch, save_path):
    r"""
    Plot the loss.
    """
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80)
    epochlist = list(range(len(train_loss_list_per_epoch)))
    _ax.plot(epochlist, train_loss_list_per_epoch, label="train")
    _ax.plot(epochlist, valid_loss_list_per_epoch, label="valid")
    _ax.set_xlabel("Epoch", fontsize=14)
    _ax.set_ylabel("Loss", fontsize=14)
    _ax.legend(loc="best")
    _plot_file_name = os.path.join(save_path, "loss.png")
    plt.savefig(_plot_file_name)
    print(f"Loss plot saved at {_plot_file_name}")
    plt.close()
