"""
A logger module which logs the training information to wandb.
NOTE: wandb is not necessary for this project.
"""

import wandb
from .train_logger import BaseLogger


class WandBLogger(BaseLogger):
    """
    A class to log the training information to wandb.
    """

    def __init__(self, project: str, entity: str, input_d: dict):
        wandb.init(project=project, entity=entity, config=input_d)

    def log_scalar(self, scalars: dict, step: int, epoch: int):
        wandb.log(scalars)

    def close(self):
        wandb.finish()
