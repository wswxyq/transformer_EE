import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .string_conv import string_to_float_list


class Pandas_NC_Dataset(Dataset):
    """
    a customized Dataset for NC dataset in DUNE
    """

    def __init__(self, config: dict):
        self.df = pd.read_csv(config["data_path"])

        for particle_feature in config["vector"]:
            self.df[particle_feature] = self.df[particle_feature].apply(
                string_to_float_list
            )

        self.len = len(self.df)
        self.maxpronglen = config["max_num_prongs"]
        self.vectornames = config["vector"]
        self.scalarnames = config["scalar"]
        self.targetname = config["target"]

        # calculate mean and std for normalization
        self.stat_scalar = []
        for x in self.scalarnames:
            self.stat_scalar.append([self.df[x].mean(), self.df[x].std()])
        self.stat_scalar = torch.Tensor(self.stat_scalar).T
        self.stat_scalar = self.stat_scalar[:, None, :]

        self.stat_vector = []
        for x in self.vectornames:
            _tmp = []
            for y in self.df[x]:
                _tmp.extend(y)
            self.stat_vector.append([np.mean(_tmp), np.std(_tmp) + 1e-5])
        self.stat_vector = torch.Tensor(self.stat_vector).T
        self.stat_vector = self.stat_vector[:, None, :]
        self.d = {}

    def __getitem__(self, index):
        if index in self.d:
            return self.d[index]

        row = self.df.iloc[index]
        _vectorsize = len(row[self.vectornames[0]])
        _vector = torch.Tensor(row[self.vectornames]).T
        _scalar = torch.Tensor(row[self.scalarnames]).T
        _vector = (_vector - self.stat_vector[0]) / self.stat_vector[1]  # modify here
        _scalar = (_scalar - self.stat_scalar[0]) / self.stat_scalar[1]  # modify here
        return_tuple = (
            # pad the vector to maxpronglen with zeros
            F.pad(_vector, (0, 0, 0, self.maxpronglen - _vectorsize), "constant", 0),
            # return the scalar
            _scalar,
            # return src_key_padding_mask
            F.pad(
                torch.zeros(_vectorsize, dtype=torch.bool),
                (0, self.maxpronglen - _vectorsize),
                "constant",
                1,  # pad with True
            ),
            torch.Tensor(row[self.targetname]),
        )
        self.d[index] = return_tuple
        return return_tuple

    def __len__(self):
        return self.len
