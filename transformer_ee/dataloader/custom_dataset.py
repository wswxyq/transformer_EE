import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformer_ee.utils.weights import create_weighter

from .sequence import sequence_statistics, string_to_float_list


class Pandas_NC_Dataset(Dataset):
    """
    A customized Dataset for NC dataset in DUNE
    Currently NC samples have slice (scalar) features and prong (sequence) features
    """

    def __init__(self, config: dict):

        # read the csv file from data_path
        self.df = pd.read_csv(config["data_path"])

        # convert string to list of float
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
        self.cached = {}
        self.weighter = None
        self.reweight = False
        if config.get("weight", None):
            self.weighter = create_weighter(config, self.df)
            self.reweight = True

    def __getitem__(self, index):
        if index in self.cached:
            return self.cached[index]

        row = self.df.iloc[index]
        _vectorsize = len(row[self.vectornames[0]])
        _vector = torch.Tensor(row[self.vectornames]).T
        _scalar = torch.Tensor(row[self.scalarnames]).T
        _vector = (_vector - self.stat_vector[0]) / self.stat_vector[1]
        _vector = F.pad(
            _vector, (0, 0, 0, self.maxpronglen - _vectorsize), "constant", 0
        )
        _scalar = (_scalar - self.stat_scalar[0]) / self.stat_scalar[1]
        _weight = 1

        if not _vectorsize:
            _vector = torch.zeros((self.maxpronglen, len(self.vectornames)))
        if self.reweight:
            _weight = self.weighter.getweight(row[self.targetname][0])
        return_tuple = (
            # pad the vector to maxpronglen with zeros
            _vector,
            # return the scalar
            _scalar,
            # return src_key_padding_mask
            F.pad(
                torch.zeros(_vectorsize, dtype=torch.bool),
                (0, self.maxpronglen - _vectorsize),
                "constant",
                1,  # pad with True
            ),
            torch.Tensor(row[self.targetname]),  # return the target
            _weight,  # return the weight
        )
        self.cached[index] = return_tuple
        return return_tuple

    def __len__(self):
        return self.len
