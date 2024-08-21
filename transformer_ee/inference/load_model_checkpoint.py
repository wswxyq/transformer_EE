"""
Create a model and load the pre-trained weights from checkpoint.
"""

import json
import os
import torch
from transformer_ee.model import create_model


def load_model_checkpoint(model_dir: str):
    """
    Load a model checkpoint and return a Predictor object.

    Args:
        model_dir: The directory containing the model checkpoint.
    Return:
        An nn.Module object representing the model.
    """
    with open(os.path.join(model_dir, "input.json"), encoding="UTF-8", mode="r") as fc:
        train_config = json.load(fc)
    net = create_model(train_config)
    net.load_state_dict(
        torch.load(
            os.path.join(model_dir, "best_model.zip"),
            map_location=torch.device("cpu"),
        )
    )
    print("Model loaded from checkpoint.")
    return net
