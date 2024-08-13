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
    parameters:
        d_model: int, the number of expected features in the vector input, must be divisible by nhead
        nhead: int, the number of heads in the multiheadattention models
        dim_feedforward: int, the dimension of the feedforward network model
        dropout: float, the dropout value for transformer
        num_layers: int, the number of TransformerEncoderLayer
        linear_hidden: int, the number of hidden units in the linear layer for scalar input
        post_linear_hidden: int, the number of hidden units in the post linear layer
    """

    def __init__(self, config):
        super().__init__()
        _kwargs = config["model"]["kwargs"]
        _nhead = _kwargs.get("nhead", 4)
        if len(config["vector"]) % _nhead != 0:
            raise ValueError(
                "The length of vector must be divisible by the length of target!"
            )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=len(
                    config["vector"]
                ),  # input dimension of transformer must match number of vector variables!
                nhead=_nhead,  # d_model must be divisible by nhead!
                dim_feedforward=_kwargs.get("dim_feedforward", 2048),
                dropout=_kwargs.get("dropout", 0.1),
                batch_first=True,
            ),
            num_layers=_kwargs.get("num_layers", 6),
        )
        linear_hidden = _kwargs.get("linear_hidden", 16)
        self.linear_scalar1 = nn.Linear(len(config["scalar"]), linear_hidden)
        self.linear_scalar2 = nn.Linear(linear_hidden, linear_hidden)

        post_linear_hidden = _kwargs.get("post_linear_hidden", 24)
        self.linear1 = nn.Linear(
            len(config["vector"]) + linear_hidden, post_linear_hidden
        )
        self.linear2 = nn.Linear(post_linear_hidden, post_linear_hidden)
        self.linear3 = nn.Linear(post_linear_hidden, len(config["target"]))

    def forward(self, x, y, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)

        y = self.linear_scalar1(y)
        y = F.relu(y)
        y = self.linear_scalar2(y)
        y = F.relu(y)

        output = torch.cat((output, y), dim=1)

        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.relu(output)
        output = self.linear3(output)

        return output
