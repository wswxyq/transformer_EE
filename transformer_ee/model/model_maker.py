"""
Create models from config.
"""

# import all the necessary model class here
from .transformerEncoder import *

# add the model classes here
model_dict = {
    "Transformer_EE_MV": Transformer_EE_MV,
}


def create_model(config: dict):
    """
    Create a model from a config.
    Args:
        "config":
        {
            "model":
            {
                name: the name of model.
                kwargs: the arguments of model.
            }
            ...
        }
    Returns:
        a model
    """

    if config["model"]["name"] not in model_dict:
        raise ValueError("Unsupported model: {}".format(config["model"]["name"]))

    model = model_dict[config["model"]["name"]](config)

    print("Model created: {}".format(model))  # print the model

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "Total number of trainable parameters: {}".format(pytorch_total_params)
    )  # print the number of trainable parameters

    return model
