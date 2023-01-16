from CONSTANT import *
from tools import *
import pandas as pd
import numpy as np
from torch.utils.data import (
    TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler)

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

import torch

from torch import nn
import torch.nn.functional as F


class BasicConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicTransformer(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channel * 4, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        x = self.transformer(x)
        return F.relu(x, inplace=True)


class PositionalEncoding(nn.Module):
    # adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-np.log(10000.0) / d_model))
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


class InceptionTransformer(nn.Module):

    def __init__(self, in_channel, seq=400):
        super().__init__()

        self.seq = seq

        self.branch1 = nn.Sequential(
            BasicConv1d(in_channel, in_channel, kernel_size=1, stride=1),
            BasicConv1d(in_channel, in_channel, kernel_size=1, stride=1),
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(in_channel, in_channel, kernel_size=3, stride=1),
            BasicConv1d(in_channel, in_channel, kernel_size=3, stride=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv1d(in_channel, in_channel, kernel_size=5, stride=1),
            BasicConv1d(in_channel, in_channel, kernel_size=5, stride=1),
        )

        self.branch4 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1),
                                     nn.ReLU())

        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.position_encoder = PositionalEncoding(
            d_model=in_channel, dropout=0.2, max_len=500)

        self.transformer = BasicTransformer(in_channel=in_channel)

    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2(x)

        branch3 = self.branch3(x)

        branch4 = self.branch4(x)

        maxlen = max(branch1.shape[2], branch2.shape[2],
                     branch3.shape[2], branch4.shape[2])

        branch1 = F.pad(input=branch1, pad=(0, maxlen - branch1.shape[2]))
        branch2 = F.pad(input=branch2, pad=(0, maxlen - branch2.shape[2]))
        branch3 = F.pad(input=branch3, pad=(0, maxlen - branch3.shape[2]))
        branch4 = F.pad(input=branch4, pad=(0, maxlen - branch4.shape[2]))

        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = self.maxpool(x)

        x = x.permute(2, 0, 1)  # permute to [seq_len, batch_size, channels]

        # x = self.position_encoder(x)
        x = self.transformer(x)  # output [seq_len, batch_size, channels]
        # permute to [batch_size, channels, seq_len]
        output = x.permute(1, 2, 0)

        return output


class SignalEncoder(nn.Module):

    def __init__(self, output_size, dropout):
        super().__init__()

        self.output_size = output_size

        self.inception1 = InceptionTransformer(in_channel=1)
        self.inception2 = InceptionTransformer(in_channel=4)
        self.inception3 = InceptionTransformer(in_channel=16)

        self.fcn = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(50, self.output_size),)

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        x = torch.mean(x, 1)  # global average pooling
        output = self.fcn(x)
        return output


class EnconderDecoder(nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()

        self.output_size = output_size
        self.encoder = SignalEncoder(output_size=self.output_size, dropout=0.2)
        self.decoder = TransformerDecoderLayer(
            d_model=1, nhead=1, dropout=0.2, batch_first=True)

    def forwad(self, x, target):
        x = self.encoder(x)  # output: [batch_size, seq_len]
        # output: [batch_size, channels, seq_len]
        x = x.reshape(-1, 1, self.output_size)
        x = x.permute(0, 2, 1)  # output: [batch_size, seq_len, channels]
        output = self.decoder(target, x)

        return output
