"""
define loss function
"""

import torch

# shape of output and target: (batch_size, output_dim)
# shape of weight: (batch_size, 1)


# base loss functions
def MSE_loss(output, target, weight=None):
    """
    mean squared error loss
    """
    # Note: torch.mean() returns the mean value of all elements in the input tensor, which is a scalar value.
    if weight is None:
        return torch.mean((output - target) ** 2)
    return torch.mean(weight * (output - target) ** 2)


def MAE_loss(output, target, weight=None):
    """
    mean absolute error loss
    """
    if weight is None:
        return torch.mean(torch.abs(output - target))
    return torch.mean(weight * torch.abs(output - target))


def MAPE_loss(output, target, weight=None):
    """
    mean absolute percentage error loss, similar to L1 loss
    """
    if weight is None:
        return torch.mean(torch.abs(output - target) / target)
    return torch.mean(weight * torch.abs(output - target) / target)


# add base loss functions to loss_function dictionary
loss_function = {
    "mean squared error": MSE_loss,
    "mean absolute error": MAE_loss,
    "mean absolute percentage error": MAPE_loss,
}


# set up a base loss function by name
def get_loss_function(loss_function_name, output, target, weight=None):
    """
    get loss function
    """
    return loss_function[loss_function_name](output, target, weight)


# complex loss functions utilize base loss functions


def linear_combination_loss(output, target, weight=None, **kwargs):
    """
    linear combination of base loss functions
    coefficients, base_loss_names should have the same length, which is the number of output variables
    e.g. kwargs = {"coefficients": [0.5, 0.5], "base_loss_names": ["mean squared error", "mean absolute error"]}
    """
    if "base_loss_names" not in kwargs or "coefficients" not in kwargs:
        raise ValueError("base_loss_names and coefficients must be provided in kwargs")

    if len(kwargs["base_loss_names"]) != len(kwargs["coefficients"]):
        raise ValueError(
            "base_loss_names and coefficients must have the same length\n",
            "len(base_loss_names):",
            len(kwargs["base_loss_names"]),
            "\nlen(coefficients):",
            len(kwargs["coefficients"]),
        )

    base_loss_names = kwargs["base_loss_names"]
    coefficients = kwargs["coefficients"]
    linear_loss = 0
    for i in range(len(base_loss_names)):
        linear_loss += coefficients[i] * loss_function[base_loss_names[i]](
            output[:, i], target[:, i], torch.squeeze(weight)
        )
    return linear_loss
