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
            nn.TransformerEncoderLayer(d_model=12, nhead=12, batch_first=True),
            num_layers=6,
        )
        self.linear = nn.Linear(12, 8)
        self.linear1 = nn.Linear(8, 2)

    def forward(self, x, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)
        output = F.relu(self.linear(output))
        output = self.linear1(output)
        return output







class Transformer_EE_v2(nn.Module):
    def __init__(self):
        super(Transformer_EE_v2, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=12, nhead=12, batch_first=True),
            num_layers=6,
        )
        self.linear = nn.Linear(12, 10)

        self.linear1 = nn.Linear(13, 8)
        self.linear2 = nn.Linear(8, 2)

    def forward(self, x, y, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)
        output = self.linear(output) # (batch_size, 12)

        output = torch.cat((output, torch.squeeze(y)), dim=1) # (batch_size, 15)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        return output







class Transformer_EE_v3(nn.Module):

    def __init__(self):
        super(Transformer_EE_v3, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=12, nhead=12, batch_first=True),
            num_layers=6,
        )
        self.linear = nn.Linear(16, 2)

        self.linear1 = nn.Linear(3, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 4)

    def forward(self, x, y, mask):
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.masked_fill(torch.unsqueeze(mask, -1), 0)
        output = torch.sum(output, dim=1)

        output1 = self.linear1(y)
        output1 = F.relu(output1)
        output1 = self.linear2(output1)
        output1 = F.relu(output1)
        output1 = self.linear3(output1)
        output1 = F.relu(output1)
        output = torch.cat((output, torch.squeeze(output1)), dim=1) # (batch_size, 16)
        output = self.linear(output)
        return output