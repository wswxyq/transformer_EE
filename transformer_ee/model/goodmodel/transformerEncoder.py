import torch
import torch.nn as nn
import torch.nn.functional as F

# by default, the input is of shape (batch_size, seq_len, embedding_dim)


class Transformer_EE_v1(nn.Module):
    """
    Only vector variable is used in this version.
    """

    def __init__(self):
        super(Transformer_EE_v1, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=12, nhead=4, batch_first=True),
            num_layers=6,
        )
        self.linear = nn.Linear(12, 2)

    def forward(self, x, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)
        output = self.linear(output)
        return output


