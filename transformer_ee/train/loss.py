"""
define loss function
"""
import torch

def MSE_loss(output, target):
    """
    mean squared error loss
    """
    return torch.mean((output - target) ** 2)

def MAE_loss(output, target):
    """
    mean absolute error loss
    """
    return torch.mean(torch.abs(output - target))

def MAPE_loss(output, target):
    """
    mean absolute percentage error loss
    """
    return torch.mean(torch.abs((output - target) / target))

loss_function = {
    "mean squared error": MSE_loss,
    "mean absolute error": MAE_loss,
    "mean absolute percentage error": MAPE_loss,
}

def get_loss_function(loss_function_name, output, target):
    """
    get loss function
    """
    return loss_function[loss_function_name](output, target)
    