""" Get information about the device """

import torch


def get_gpu():
    """
    Get information about the GPU
    """

    _device = torch.device("cpu")

    if torch.cuda.is_available():
        _device = torch.device("cuda")
        print("Using NVIDIA GPU")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
        print("Using Apple MPS")
    else:
        print("Using CPU")
    return _device
