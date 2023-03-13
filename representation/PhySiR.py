# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2022-12-30 18:43:38

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class BasicConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

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
        self.branch4 = nn.Sequential(
            BasicConv1d(in_channel, in_channel, kernel_size=7, stride=1),
            BasicConv1d(in_channel, in_channel, kernel_size=7, stride=1),
        )
        self.branch5 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1),
                                     nn.ReLU())

    def forward(self, x):
        branch1 = self.branch1(x)
        branch1 = F.pad(input=branch1, pad=(0, self.seq - branch1.shape[2]))

        branch2 = self.branch2(x)
        branch2 = F.pad(input=branch2, pad=(0, self.seq - branch2.shape[2]))

        branch3 = self.branch3(x)
        branch3 = F.pad(input=branch3, pad=(0, self.seq - branch3.shape[2]))

        branch4 = self.branch4(x)
        branch4 = F.pad(input=branch4, pad=(0, self.seq - branch4.shape[2]))

        branch5 = self.branch5(x)
        branch5 = F.pad(input=branch5, pad=(0, self.seq - branch5.shape[2]))

        outputs = [branch1, branch2, branch3, branch4, branch5]

        return torch.cat(outputs, 1)


class SignalEncoder(nn.Module):

    def __init__(self, output_size, dropout, seq=400):
        super().__init__()

        self.seq = seq

        self.output_size = output_size

        self.inception1 = Inception(in_channel=1)
        self.inception2 = Inception(in_channel=5)
        self.inception3 = Inception(in_channel=25)
        self.inception4 = Inception(in_channel=125)

        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.fcn = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(seq, self.output_size))

    def forward(self, x):
        x = self.inception1(x)
        x = self.maxpool1(x)

        x = self.inception2(x)
        x = self.maxpool2(x)

        x = self.inception3(x)
        x = self.maxpool3(x)

        x = self.inception4(x)
        # x = self.maxpool4(x)

        output, _ = torch.max(x, 1)  # global max pooling

        return self.fcn(output)


class SignalEmbedding(nn.Module):

    def __init__(self, output_size, dropout=0.2, seq=400):
        super().__init__()

        self.seq = seq

        self.bvp_encoder = SignalEncoder(output_size, dropout)
        self.eda_encoder = SignalEncoder(output_size, dropout)
        self.temp_encoder = SignalEncoder(output_size, dropout)
        self.hr_encoder = SignalEncoder(output_size, dropout)

    def forward(self, x):

        bvp = x[:, 0, :].reshape(-1, 1, self.seq)
        eda = x[:, 1, :].reshape(-1, 1, self.seq)
        temp = x[:, 2, :].reshape(-1, 1, self.seq)
        hr = x[:, 3, :].reshape(-1, 1, self.seq)

        bvp_encoder = self.bvp_encoder(bvp)
        eda_encoder = self.eda_encoder(eda)
        temp_encoder = self.temp_encoder(temp)
        hr_encoder = self.hr_encoder(hr)

        outputs = [bvp_encoder, eda_encoder, temp_encoder, hr_encoder]

        return torch.stack(outputs, 1)


class SigRepSimple(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.output_size = 40
        # self.n_class = n_class

        self.signal_embedd = SignalEmbedding(output_size=self.output_size,
                                             dropout=args.dropout)

        self.fcn = nn.Sequential(nn.Linear(self.output_size * 4, 16),
                                 nn.ReLU(), nn.Linear(16, 8), nn.ReLU(),
                                 nn.Dropout(p=args.dropout))

        self.regressor = nn.Linear(8, 1)

    def forward(self, x):
        x = self.signal_embedd(x)

        x = x.flatten(start_dim=1)
        x = self.fcn(x)
        output = self.regressor(x)
        return output
