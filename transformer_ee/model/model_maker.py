"""
Create models from config.
"""
# import all the necessary model class here
from .transformerEncoder import *

# add the model classes here
model_dict = {
    "Transformer_EE_v1": Transformer_EE_v1,
    "Transformer_EE_v2": Transformer_EE_v2,
    "Transformer_EE_v3": Transformer_EE_v3,
    "Transformer_EE_v4": Transformer_EE_v4,
    "Transformer_EE_v5": Transformer_EE_v5,
    "Transformer_EE_v6": Transformer_EE_v6,
    "Transformer_EE_v7": Transformer_EE_v7,
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
    _kwgs = config["model"]["kwargs"]
    if config["model"]["name"] not in model_dict:
        raise ValueError("Unsupported model: {}".format(config["model"]["name"]))

    model = model_dict[config["model"]["name"]](**_kwgs)

    print("Model created: {}".format(model))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of trainable parameters: {}".format(pytorch_total_params))

    return model
