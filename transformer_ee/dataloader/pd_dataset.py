import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformer_ee.utils.weights import create_weighter

from .sequence import sequence_statistics, string_to_float_list


class pandas_Dataset(Dataset):  # pylint: disable=C0103, W0223
    """
    A base PyTorch dataset for pandas dataframe
    """

    def __init__(
        self, config: dict, dtframe: pd.DataFrame, weighter=None, eval=False
    ):  # pylint: disable=W0622
        self.eval = eval
        self.config = config.copy()

        self.df = dtframe

        self.maxpronglen = config["max_num_prongs"]
        self.vectornames = config["vector"]
        self.scalarnames = config["scalar"]
        self.targetname = config["target"] if not self.eval else None
        # If weighter is not provided, create a new one from config and dataframe.
        # For prediction, the weighter should be set to NullWeights() manually.
        self.weighter = None
        if self.eval:
            print("In evaluation mode, target = 0 and weighter = 1.")
        elif weighter is None:
            self.weighter = create_weighter(self.config, self.df)
            print("Created weighter from config and data. Type: ", type(self.weighter))
        else:
            self.weighter = weighter
            print("Using provided weighter. Type: ", type(self.weighter))

        # convert string to list of float
        for sequence_name in self.config["vector"]:
            self.df[sequence_name] = self.df[sequence_name].apply(string_to_float_list)

    def __len__(self):
        return len(self.df)


class Normalized_pandas_Dataset_with_cache(pandas_Dataset):
    """
    A base PyTorch dataset for pandas dataframe with normalization and caching
    """

    def __init__(
        self,
        config: dict,
        dtframe: pd.DataFrame,
        weighter=None,
        eval=False,  # pylint: disable=W0622
        use_cache=True,
    ):
        super().__init__(config, dtframe, weighter=weighter, eval=eval)
        self.use_cache = use_cache
        self.cached = {}
        self.normalized_df = self.df.copy()

        # convert list of float to numpy array
        for name in self.vectornames:
            self.normalized_df[name] = self.normalized_df[name].apply(np.array)
        self.stat = {}
        self.normalized = False

    def statistic(self):
        """
        Calculate the mean and standard deviation with respect to each column.
        """
        if self.eval:
            raise ValueError("In evaluation mode, do not call statistic()!")
        if self.normalized:
            raise ValueError("Already normalized! Do not call statistic() again!")

        # calculate mean and std for sequence features
        for sequence_name in self.vectornames:
            self.stat[sequence_name] = sequence_statistics(self.df[sequence_name])

        # calculate mean and std for scalar features
        for scalar_name in self.scalarnames:
            self.stat[scalar_name] = [
                np.mean(self.df[scalar_name]),
                np.std(self.df[scalar_name]) + 1e-10,
            ]

    def normalize(self, stat=None):
        """
        Normalize the dataset with respect to the provided statistics.
        If no statistics is provided, use the statistics calculated by statistic().
        """

        _stat = stat
        if _stat is None:  # by default, use the statistics calculated by statistic()
            if self.eval:
                raise ValueError("In evaluation mode, stat cannot be None!")
            print("Using statistics calculated by statistic()!")
            if not self.stat:
                raise ValueError("Please call statistics() first!")
            _stat = self.stat

        if self.normalized:
            raise ValueError("Already normalized! Do not call normalize() again!")

        for sequence_name in self.vectornames + self.scalarnames:
            self.normalized_df[sequence_name] = self.normalized_df[sequence_name].apply(
                lambda x, seq_name: (x - _stat[seq_name][0]) / _stat[seq_name][1],
                args=(sequence_name,),
            )

        self.normalized = True

    def __getitem__(self, index):
        if index in self.cached:
            return self.cached[index]

        row = self.normalized_df.iloc[index]  # get the row

        if not self.normalized:
            raise ValueError("Please call normalize() first!")

        _vectorsize = len(row[self.vectornames[0]])
        _vector = torch.Tensor(np.stack(row[self.vectornames].values))
        _scalar = torch.Tensor(np.stack(row[self.scalarnames].values))

        _vector = _vector.T

        _mask = torch.Tensor([0] * _vectorsize)

        if _vectorsize < self.maxpronglen:
            _vector = F.pad(
                _vector, (0, 0, 0, self.maxpronglen - _vectorsize), "constant", 0
            )
            _mask = F.pad(_mask, (0, self.maxpronglen - _vectorsize), "constant", 1)
        else:
            _vector = _vector[: self.maxpronglen, :]
            _mask = _mask[: self.maxpronglen]

        if not _vectorsize:
            _vector = torch.zeros((self.maxpronglen, len(self.vectornames)))
            _mask = torch.ones(self.maxpronglen)

        _target = 0.0
        _weight = 1.0
        if not self.eval:
            _target = torch.Tensor(np.stack(row[self.targetname].values))
            _weight = torch.Tensor([self.weighter.getweight(row[self.targetname[0]])])

        return_tuple = (
            _vector,  # shape: (max_seq_len, vector_dim)
            _scalar,  # shape: (scalar_dim)
            _mask.to(torch.bool),  # shape: (max_seq_len)
            _target,  # shape: (target_dim)
            _weight,  # shape: (1)
        )
        if self.use_cache:
            self.cached[index] = return_tuple
        return return_tuple
