"""
A callable class to add noise to the input data of model during training.
Please read the following note before using this class.

Why callable?
- Because collate_fn in DataLoader should be callable.

What exactly does this class do?
- This class performs a LINEAR TRANSFORMATION on the input data, which is equivalent to:
    1. Multiplying the un-normalized input data by a random number centered at 1.
    2. Normalizing it. (transformer_EE always normalizes the input data)

In transformer_EE, the input data is normalized and cached to avoid repeating the normalization 
process. This makes adding noise to the input data a bit tricky if we want the noise to be different 
for each epoch. Thus, we need to trace back the normalization process and add noise to the 
normalized input data.

Assuming we intend to perform the following transformation on the un-normalized input:
    x -> x * (1 + noise)
and x is normalized to x' by the following formula:
    x' = (x - mean) / std
where mean and std are the mean and standard deviation of the training set, respectively.
Reverse the normalization process to get x:
    x = x' * std + mean
then we have:
    x' -> (x * (1 + noise) - mean) / std
        = ((x' * std + mean) * (1 + noise) - mean) / std
        = x' * (1 + noise) + noise * mean / std
"""

import torch
from torch.utils.data._utils.collate import default_collate


class normalized_noise:
    """
    Use this with Normalized_pandas_Dataset_with_cache.
    """

    def __init__(self, config, stats):
        """
        config:     the configuration dictionary
        stats:      the statistics of the training set; a dictionary with keys
                    "vector" and "scalar" containing the mean and std of the training
        example of config:
        {
            "vector": ["sequence1", "sequence2"],
            "scalar": ["scalar1", "scalar2"],
            "noise": {
                "name": "gaussian",
                "loc": 0,
                "scale": 0.1,
                "vector": ["sequence1"],
                "scalar": ["scalar1"],
            }
        }
        """
        if "noise" not in config:
            raise ValueError("Noise configuration not found!")
        self.noise = None
        if config["noise"]["name"] == "gaussian":
            self.noise = lambda size: torch.normal(
                mean=config["noise"]["mean"], std=config["noise"]["std"], size=size
            )
        elif config["noise"]["name"] == "uniform":
            self.noise = lambda size: config["noise"]["low"] + torch.rand(size=size) * (
                config["noise"]["high"] - config["noise"]["low"]
            )
        else:
            raise NotImplementedError("Noise distribution not implemented!")

        # get indices of vector, scalar that need to be added noise
        self.vector_indices = [
            config["vector"].index(name) for name in config["noise"]["vector"]
        ]
        self.scalar_indices = [
            config["scalar"].index(name) for name in config["noise"]["scalar"]
        ]
        # find the indices of the vector and scalar names
        # calculate mean / std for the vector and scalar
        self.vector_name_indices = []
        self.scalar_name_indices = []
        self.vector_correction = []
        self.scalar_correction = []
        for name in config["noise"].get("vector", []):
            self.vector_name_indices.append(config["vector"].index(name))
            self.vector_correction.append(stats[name][0] / stats[name][1])
        for name in config["noise"].get("scalar", []):
            self.scalar_name_indices.append(config["scalar"].index(name))
            self.scalar_correction.append(stats[name][0] / stats[name][1])
        self.vector_correction = torch.Tensor(self.vector_correction)
        self.scalar_correction = torch.Tensor(self.scalar_correction)

    def __call__(self, batch):
        """
        batch: the normalized input data
        return: a tuple of noisy input data and target data
        """

        # default_collate to convert the list of tensors to a tensor
        batch = default_collate(batch)
        vector, scalar, mask, target, weight = batch
        # shape of vector: (batch_size, max_seq_len, vector_dim)
        # shape of scalar: (batch_size, scalar_dim)
        # shape of mask: (batch_size, max_seq_len)
        # shape of target: (batch_size, target_dim)
        # shape of weight: (batch_size, 1)
        _noise = self.noise(size=(vector.shape[0],))
        # apply noise to the vector
        if self.vector_indices:
            vector[:, :, self.vector_indices] += (
                vector[:, :, self.vector_indices] * _noise[:, None, None]
                + self.vector_correction[None, None, :] * _noise[:, None, None]
            )
        # apply noise to the scalar
        if self.scalar_indices:
            scalar[:, self.scalar_indices] += (
                scalar[:, self.scalar_indices] * _noise[:, None]
                + self.scalar_correction[None, :] * _noise[:, None]
            )
        # NOTE: after adding noise, the padding values may not be 0 anymore,
        # but it should not matter
        return vector, scalar, mask, target, weight
