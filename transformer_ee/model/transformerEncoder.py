"""
Transformer Encoder based energy estimator models.
"""

import torch
from torch import nn

# by default, the input is of shape (batch_size, seq_len, embedding_dim)


class Transformer_EE_v1(nn.Module):
    """
    Only vector variable is used in this version.
    """

    def __init__(self):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=12, nhead=4, batch_first=True),
            num_layers=6,
        )
        self.linear = nn.Linear(12, 2)

    def forward(self, x, y, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)
        output = self.linear(output)
        return output


class Transformer_EE_v2(nn.Module):
    """
    Only vector variable is used in this version.

    Make a stacked transformer encoder.
    """

    def __init__(self):
        super().__init__()
        self.transformer_encoder_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=12, nhead=4, batch_first=True),
            num_layers=1,
        )

        self.linear_1 = nn.Linear(12, 48)

        self.transformer_encoder_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=48, nhead=8, batch_first=True),
            num_layers=1,
        )

        self.linear_2 = nn.Linear(48, 256)

        self.transformer_encoder_3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=1,
        )

        self.linear_3 = nn.Linear(256, 1024)

        self.transformer_encoder_4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True),
            num_layers=1,
        )

        self.linear_4 = nn.Linear(1024, 2)

    def forward(self, x, mask):
        output = self.transformer_encoder_1(x, src_key_padding_mask=mask)
        output = self.linear_1(output)

        output = self.transformer_encoder_2(output, src_key_padding_mask=mask)
        output = self.linear_2(output)

        output = self.transformer_encoder_3(output, src_key_padding_mask=mask)
        output = self.linear_3(output)

        output = self.transformer_encoder_4(output, src_key_padding_mask=mask)

        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)

        output = torch.sum(output, dim=1)
        output = self.linear_4(output)
        return output
