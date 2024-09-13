import io
import lzma

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformer_ee.utils.weights import create_weighter


def get_polars_df_from_xz_file(file_path):
    """
    Reads an xz file and returns a Polars DataFrame.
    """

    with open(file_path, "rb") as f:
        xz_data = f.read()
    memory_file = io.BytesIO(xz_data)
    print("Decompressing data from xz file", file_path, "to memory...")
    decompressed_data = lzma.decompress(memory_file.read())
    decompressed_memory_file = io.BytesIO(decompressed_data)
    return pl.read_csv(decompressed_memory_file)


def get_polars_df_from_file(file_path):
    """
    Reads a file and returns a Polars DataFrame.

    Args:
        file_path (str): The path to the file.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the data from the file.

    Raises:
        ValueError: If the file format is not supported.
    """
    if file_path.endswith(".csv"):
        return pl.read_csv(file_path)
    elif file_path.endswith(".xz"):
        return get_polars_df_from_xz_file(file_path)
    else:
        raise ValueError("File format not supported")


class Polars_Dataset(Dataset): # pylint: disable=C0103, W0223
    """
    A base PyTorch dataset for polars dataframe
    """

    def __init__(self, config: dict, dtframe: pl.DataFrame, weighter=None, eval=False): # pylint: disable=W0622
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
            self.weighter = create_weighter(
                self.config, self.df[self.targetname[0]].to_numpy()
            )
            print("Created weighter from config and data. Type: ", type(self.weighter))
        else:
            self.weighter = weighter
            print("Using provided weighter. Type: ", type(self.weighter))

        # convert string to list of float
        for sequence_name in self.config["vector"]:
            self.df = self.df.with_columns(
                self.df.get_column(sequence_name).replace("", "0")
            )
            self.df = self.df.with_columns(
                self.df.get_column(sequence_name)
                .str.split(by=",")
                .cast(pl.List(pl.Float32))
            )

    def __len__(self):
        return len(self.df)


class Normalized_Polars_Dataset_with_cache(Polars_Dataset): # pylint: disable=C0103
    """
    A base PyTorch dataset for polars dataframe with normalization and caching.
    NOTE: Normalization is done in place.
    """

    def __init__(
        self,
        config: dict,
        dtframe: pl.DataFrame,
        weighter=None,
        eval=False,
        use_cache=True,
    ):
        super().__init__(config, dtframe, weighter=weighter, eval=eval)
        self.use_cache = use_cache
        self.cached = {}
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
            _ = self.df.get_column(sequence_name).explode()
            self.stat[sequence_name] = [_.mean(), _.std() + 1e-10]
            del _

        # calculate mean and std for scalar features
        for scalar_name in self.scalarnames:
            self.stat[scalar_name] = [
                self.df.get_column(scalar_name).mean(),
                self.df.get_column(scalar_name).std() + 1e-10,
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

        for sc_name in self.scalarnames:
            self.df = self.df.with_columns(
                (self.df.select(sc_name) - _stat[sc_name][0]) / _stat[sc_name][1]
            )

        for sequence_name in self.vectornames:
            self.df = self.df.with_columns(
                self.df.get_column(sequence_name).list.eval(
                    (pl.element() - _stat[sequence_name][0]) / _stat[sequence_name][1]
                )
            )

        self.normalized = True

    def __getitem__(self, index):
        if index in self.cached:
            return self.cached[index]

        if not self.normalized:
            raise ValueError("Please call normalize() first!")

        _vectorsize = len(self.df[index, self.vectornames[0]])
        _vector = torch.Tensor(
            [self.df[index, v].to_list() for v in self.vectornames]
        ).T
        _scalar = torch.Tensor([self.df[index, s] for s in self.scalarnames])

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
            _target = torch.Tensor([self.df[index, t] for t in self.targetname])
            _weight = torch.Tensor(
                [self.weighter.getweight(self.df[index, self.targetname[0]])]
            )

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
