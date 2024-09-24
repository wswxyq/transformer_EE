"""
Export the model to TorchScript.
"""

import json
import os

import torch
import torch.nn as nn

from .load_model_checkpoint import load_model_checkpoint


class ModelWithNormalization(nn.Module):
    """
    A wrapper class for the model with normalization.
    """

    def __init__(self, train_config: dict, train_stat: dict, ee_model: nn.Module):
        super().__init__()
        self.train_config = train_config
        self.train_stat = train_stat
        self.ee_model = ee_model
        self.vector_mean = torch.zeros(len(train_config["vector"]))
        self.vector_std = torch.zeros(len(train_config["vector"]))
        self.scalar_mean = torch.zeros(len(train_config["scalar"]))
        self.scalar_std = torch.zeros(len(train_config["scalar"]))
        for i, name in enumerate(train_config["vector"]):
            self.vector_mean[i] = train_stat[name][0]
            self.vector_std[i] = train_stat[name][1]
        for i, name in enumerate(train_config["scalar"]):
            self.scalar_mean[i] = train_stat[name][0]
            self.scalar_std[i] = train_stat[name][1]
        # Visit https://github.com/pytorch/pytorch/issues/12810 for use of None in indexing
        self.vector_mean = self.vector_mean[None, None, :]
        self.vector_std = self.vector_std[None, None, :]
        self.scalar_mean = self.scalar_mean[None, :]
        self.scalar_std = self.scalar_std[None, :]

    def forward(self, vector, scalar):
        return self.ee_model(
            (vector - self.vector_mean) / self.vector_std,
            (scalar - self.scalar_mean) / self.scalar_std,
            torch.zeros(vector.shape[0], vector.shape[1], dtype=torch.bool),
        )


def export_model(model_dir: str):
    """
    Export the model to TorchScript.
    User should prepare the normalzed input data for the model.

    Args:
        model_dir: The directory containing the model checkpoint.
    """
    net = load_model_checkpoint(model_dir)
    net = net.cpu()  # Export to CPU
    net.eval()
    script_model = torch.jit.script(net)
    script_model.save(os.path.join(model_dir, "cpu_model.pt"))
    print("Model without input normalization exported to TorchScript.")


def export_model_with_normalization(model_dir: str):
    """
    Add a normalization layer to the model and export it to TorchScript.
    The output model expects the raw input data.

    Args:
        model_dir: The directory containing the model checkpoint.
    """
    with open(os.path.join(model_dir, "input.json"), encoding="UTF-8", mode="r") as fc:
        train_config = json.load(fc)
    with open(
        os.path.join(model_dir, "trainset_stat.json"), encoding="UTF-8", mode="r"
    ) as fc:
        train_stat = json.load(fc)
    net = load_model_checkpoint(model_dir)
    net = net.cpu()  # Export to CPU
    net.eval()
    norm_model = ModelWithNormalization(train_config, train_stat, net)
    script_model = torch.jit.script(norm_model)
    script_model.save(os.path.join(model_dir, "cpu_model_norm.pt"))
    print("Model with input normalization exported to TorchScript.")
