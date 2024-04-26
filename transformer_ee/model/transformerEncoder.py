"""
Transformer Encoder based energy estimator models.
"""

import torch
from torch import nn
from torch.nn import functional as F

# by default, the input is of shape (batch_size, seq_len, embedding_dim)


class Transformer_EE_MV(nn.Module):
    """
    information of slice and prongs are cancatenated together.
    """

    def __init__(self, config):
        super().__init__()
        if len(config["vector"])%len(config["target"]) != 0:
            raise ValueError("The length of vector must be divisible by the length of target!")
        _kwargs = config["model"]["kwargs"]
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=len(config["vector"]), # input dimension of transformer must match number of vector variables!
                nhead=_kwargs.get("nhead", 4), # d_model must be divisible by nhead!
                dim_feedforward=_kwargs.get("dim_feedforward", 2048),
                dropout=_kwargs.get("dropout", 0.1),
                batch_first=True,
            ),
            num_layers=_kwargs.get("num_layers", 6),
        )
        self.linear_scalar1 = nn.Linear(3, 6)
        self.linear_scalar2 = nn.Linear(6, 12)

        self.linear1 = nn.Linear(24, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, len(config["target"]))

    def forward(self, x, y, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)

        y = self.linear_scalar1(y)
        y = F.relu(y)
        y = self.linear_scalar2(y)
        y = F.relu(y)

        output = torch.cat((output, torch.squeeze(y)), 1)

        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.relu(output)
        output = self.linear3(output)

        return output
