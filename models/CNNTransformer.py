import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    # adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pos_encoder = PositionalEncoding(self.args.out_channels, args.dropout)
        self.encoder_layers = TransformerEncoderLayer(d_model=self.args.out_channels, nhead=self.args.nhead, dropout=args.dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, self.args.nlayers, norm=None)
        self.decoder = nn.Linear(self.args.out_channels, self.args.out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, embedding_dim]
            No available: src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class CTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cnns = nn.Sequential(
            nn.Conv1d(4, 16, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, args.out_channels, 3),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(args.out_channels),
            nn.ReLU(),
        )
        
        self.transformer = TransformerBlock(args)

        self.fcn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(args.fcn_input, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        input [batch_size, channels, seq_len]
        """
        x = self.cnns(x)  # output [batch_size, channels, seq_len]
        
        x = x.permute(2, 0, 1) # permute to [seq_len, batch_size, channels]
        x = self.transformer(x) # output [seq_len, batch_size, channels]
        x = x.permute(1, 0, 2) # permute to [batch_size, seq_len, channels]
        x = F.relu(x)
        
        x = x.flatten(start_dim=1) # flatten to [batch_size, seq_len * channels]
        output = self.fcn(x)
        return output
