""" Get information about the device """

import torch


def get_gpu():

    """
    Get information about the GPU
    """

    _mps_device = torch.device("cpu")

    if torch.cuda.is_available():
        _mps_device = torch.device("cuda")
        print("Using NVIDIA GPU")
    else:
        if torch.backends.mps.is_available():
            _mps_device = torch.device("mps")
            print("Using Apple MPS")
    return _mps_device
