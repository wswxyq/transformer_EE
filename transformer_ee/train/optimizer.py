"""
A function to create an optimizer from a config.
"""

import torch.optim as optim
import torch.nn as nn


def create_optimizer(config: dict, model: nn.Module):
    """
    Create an optimizer from a config.
    Args:
        "config":
        {
            "optimizer":
            {
                name: the name of optimizer.
                kwargs: the arguments of optimizer.
            ...
            }
        }

        model: a model of nn.Module class to optimize
    Returns:
        an optimizer
    """
    _kwgs = config["optimizer"]["kwargs"]
    if config["optimizer"]["name"] == "Adam":
        optimizer = optim.Adam(model.parameters(), **_kwgs)
    elif config["optimizer"]["name"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), **_kwgs)
    elif config["optimizer"]["name"] == "adamax":
        optimizer = optim.Adamax(model.parameters(), **_kwgs)
    elif config["optimizer"]["name"] == "sgd":
        optimizer = optim.SGD(model.parameters(), **_kwgs)
    elif config["optimizer"]["name"] == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), **_kwgs)
    else:
        raise ValueError("Unsupported optimizer: {}".format(config.optimizer))

    return optimizer
