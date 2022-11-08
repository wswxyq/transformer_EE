"""
Load the data
"""

import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset


def split_data(dataset: Dataset, config):
    """
    A function to split the data into train, validation and test sets
    """

    batch_size_train = config["batch_size_train"]
    batch_size_valid = config["batch_size_valid"]
    batch_size_test = config["batch_size_test"]

    seed = config["seed"]

    _indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(_indices)

    test_size = config["test_size"]
    valid_size = config["valid_size"]

    if isinstance(test_size, float):
        test_size = int(len(_indices) * test_size)
    if isinstance(valid_size, float):
        valid_size = int(len(_indices) * valid_size)

    train_indicies = _indices[: len(_indices) - valid_size - test_size]
    valid_indicies = _indices[
        len(_indices) - valid_size - test_size : len(_indices) - test_size
    ]
    test_indicies = _indices[len(_indices) - test_size :]

    train_dataset = Subset(dataset, train_indicies)
    valid_dataset = Subset(dataset, valid_indicies)
    test_dataset = Subset(dataset, test_indicies)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True
    )

    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size_valid, shuffle=False
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False
    )

    return trainloader, validloader, testloader
