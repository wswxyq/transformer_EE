"""
Export the model to TorchScript.
"""

import os
import torch
from .load_model_checkpoint import load_model_checkpoint


def export_model(model_dir: str):
    """
    Export the model to TorchScript.

    Args:
        model_dir: The directory containing the model checkpoint.
    """
    net = load_model_checkpoint(model_dir)
    net = net.cpu()  # Export to CPU
    net.eval()
    script_model = torch.jit.script(net)
    script_model.save(os.path.join(model_dir, "cpu_model.pt"))
    print("Model exported to TorchScript.")
