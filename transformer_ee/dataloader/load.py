"""
Load the data
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from transformer_ee.dataloader.pd_dataset import Normalized_pandas_Dataset_with_cache


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


def get_sample_indices(sample_size: int, config) -> tuple:
    """
    A function to get the indices of the samples

    sample_size:    the number of samples
    config:         the configuration dictionary
    return:         the indices of train, validation and test sets
    """

    seed = config["seed"]

    _indices = np.arange(sample_size)
    np.random.seed(seed)
    np.random.shuffle(_indices)

    test_size = config["test_size"]
    valid_size = config["valid_size"]

    # test_size and valid_size can be either int or float
    if isinstance(test_size, float):
        test_size = int(sample_size * test_size)
    if isinstance(valid_size, float):
        valid_size = int(sample_size * valid_size)

    train_indicies = _indices[: sample_size - valid_size - test_size]
    valid_indicies = _indices[
        sample_size - valid_size - test_size : sample_size - test_size
    ]
    test_indicies = _indices[sample_size - test_size :]

    print("train indicies size:\t", len(train_indicies))
    print("valid indicies size:\t", len(valid_indicies))
    print("test  indicies size:\t", len(test_indicies))

    return train_indicies, valid_indicies, test_indicies


def get_train_valid_test_dataloader(config: dict):
    """
    A function to get the train, validation and test datasets
    Use the statistic of the training set to normalize the validation and test sets
    """
    df = pd.read_csv(config["data_path"])
    train_idx, valid_idx, test_idx = get_sample_indices(len(df), config)
    train_set = Normalized_pandas_Dataset_with_cache(
        config, df.iloc[train_idx].reset_index(drop=True, inplace=False)
    )
    valid_set = Normalized_pandas_Dataset_with_cache(
        config, df.iloc[valid_idx].reset_index(drop=True, inplace=False)
    )
    test_set = Normalized_pandas_Dataset_with_cache(
        config, df.iloc[test_idx].reset_index(drop=True, inplace=False)
    )

    train_set.statistic()

    train_set.normalize()
    valid_set.normalize(train_set.stat)
    test_set.normalize(train_set.stat)

    batch_size_train = config["batch_size_train"]
    batch_size_valid = config["batch_size_valid"]
    batch_size_test = config["batch_size_test"]

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_train, shuffle=True
    )

    validloader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size_valid, shuffle=False
    )

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False
    )

    return trainloader, validloader, testloader
