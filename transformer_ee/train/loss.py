"""
define loss function
"""
import torch


def MSE_loss(output, target, weight=None):
    """
    mean squared error loss
    """
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


loss_function = {
    "mean squared error": MSE_loss,
    "mean absolute error": MAE_loss,
    "mean absolute percentage error": MAPE_loss,
}


def get_loss_function(loss_function_name, output, target, weight=None):
    """
    get loss function
    """
    return loss_function[loss_function_name](output, target, weight)
