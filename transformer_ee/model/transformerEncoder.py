"""
Transformer Encoder based energy estimator models.
"""

import torch
from torch import nn
from torch.nn import functional as F

# by default, the input is of shape (batch_size, seq_len, embedding_dim)


class Transformer_EE_v1(nn.Module):
    """
    Only vector variable is used in this version.
    """

    def __init__(self, num_layers=6, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=12,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.linear = nn.Linear(12, 2)

    def forward(self, x, y, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)
        output = self.linear(output)
        return output


class Transformer_EE_v2(nn.Module):
    # stacked transformer encoder
    """
    Only vector variable is used in this version.

    Make a stacked transformer encoder.
    """

    def __init__(self, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.transformer_encoder_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=12,
                nhead=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_1 = nn.Linear(12, 48)

        self.transformer_encoder_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=48,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_2 = nn.Linear(48, 256)

        self.transformer_encoder_3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_3 = nn.Linear(256, 1024)

        self.transformer_encoder_4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1024,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_4 = nn.Linear(1024, 2)

    def forward(self, x, y, mask):
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


class Transformer_EE_v3(nn.Module):
    """
    information of slice and prongs are added together.
    """

    def __init__(self, num_layers=6, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=12,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.linear_scalar1 = nn.Linear(3, 6)
        self.linear_scalar2 = nn.Linear(6, 12)

        self.linear = nn.Linear(12, 2)

    def forward(self, x, y, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)

        y = self.linear_scalar1(y)
        y = F.relu(y)
        y = self.linear_scalar2(y)
        y = F.relu(y)

        output = output + torch.squeeze(y)

        output = self.linear(output)

        return output


class Transformer_EE_v4(nn.Module):
    """
        information of slice and prongs are cancatenated together.
    """

    def __init__(self, num_layers=6, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=12,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.linear_scalar1 = nn.Linear(3, 6)
        self.linear_scalar2 = nn.Linear(6, 12)

        self.linear1 = nn.Linear(24, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, 2)

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
        

class Transformer_EE_v5(nn.Module):
    """
        information of slice and prongs are cancatenated together.
    """

    def __init__(self, num_layers=12, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.preprocess_1 = nn.Linear(12, 48)
        self.preprocess_2 = nn.Linear(48, 128)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_scalar1 = nn.Linear(3, 24)
        self.linear_scalar2 = nn.Linear(24, 64)
        self.linear_scalar3 = nn.Linear(64, 128)

        self.linear1 = nn.Linear(256, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, x, y, mask):
        output = self.preprocess_1(x) # 12 -> 48
        output = F.relu(output)
        output = self.preprocess_2(output) # 48 -> 128
        output = self.transformer_encoder(output, src_key_padding_mask=mask)
        output = torch.sum(output, dim=1) # 128 -> 128

        y = self.linear_scalar1(y) # 3 -> 24
        y = F.relu(y)
        y = self.linear_scalar2(y) # 24 -> 64
        y = F.relu(y)
        y = self.linear_scalar3(y) # 64 -> 128

        output = torch.cat((output, torch.squeeze(y)), 1) # 128 + 128 -> 256

        output = self.linear1(output) # 256 -> 32
        output = F.relu(output)
        output = self.linear2(output) # 32 -> 2

        return output

class Transformer_EE_v6(nn.Module):
    """
        information of slice and prongs are cancatenated together.
    """

    def __init__(self, num_layers=6, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.preprocess_1 = nn.Linear(12, 32)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.linear_scalar1 = nn.Linear(3, 6)
        self.linear_scalar2 = nn.Linear(6, 12)

        self.linear1 = nn.Linear(44, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, 2)

    def forward(self, x, y, mask):
        output = self.preprocess_1(x) # 12 -> 32
        output = self.transformer_encoder(output, src_key_padding_mask=mask)
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


class Transformer_EE_v7(nn.Module):
    # stacked transformer encoder
    """
    Only vector variable is used in this version.

    Make a stacked transformer encoder.
    """

    def __init__(self, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.transformer_encoder_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=12,
                nhead=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_1 = nn.Linear(12, 24)

        self.transformer_encoder_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=24,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_2 = nn.Linear(24, 48)

        self.transformer_encoder_3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=48,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_3 = nn.Linear(48, 96)

        self.transformer_encoder_4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=96,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_4 = nn.Linear(96, 192)

        self.transformer_encoder_5 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_5 = nn.Linear(192, 384)

        self.transformer_encoder_6 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=384,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.linear_6 = nn.Linear(384, 2)




    def forward(self, x, y, mask):
        output = self.transformer_encoder_1(x, src_key_padding_mask=mask)
        output = self.linear_1(output)

        output = self.transformer_encoder_2(output, src_key_padding_mask=mask)
        output = self.linear_2(output)

        output = self.transformer_encoder_3(output, src_key_padding_mask=mask)
        output = self.linear_3(output)

        output = self.transformer_encoder_4(output, src_key_padding_mask=mask)
        output = self.linear_4(output)

        output = self.transformer_encoder_5(output, src_key_padding_mask=mask)
        output = self.linear_5(output)

        output = self.transformer_encoder_6(output, src_key_padding_mask=mask)

        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)

        output = torch.sum(output, dim=1)
        output = self.linear_6(output)
        return output