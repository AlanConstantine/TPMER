# from CONSTANT import *
from tools import *
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import (
    TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler)

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

import torch
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

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
        return F.leaky_relu(x, inplace=True)


class BasicTransformer(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channel * 4, nhead=4, activation='gelu')
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=2)
        self.ln = nn.LayerNorm(in_channel * 4)

        self.position_encoder = PositionalEncoding(
            d_model=in_channel * 4, dropout=0.1, max_len=768)

    def forward(self, x):
        x = self.position_encoder(x)
        x = self.transformer(x)
        x = self.ln(x)
        return F.gelu(x)


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
                                     nn.LeakyReLU())

        self.maxpool = nn.MaxPool1d(kernel_size=2)

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

        # x = self.position_encoder(x)  # ablation experiment
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
        # self.inception4 = InceptionTransformer(in_channel=64)

        self.fcn = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(96, self.output_size),)

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        # x = self.inception4(x)

        x = torch.mean(x, 1)  # global average pooling
        output = self.fcn(x)
        return output


class MultiSignalEncoder(nn.Module):
    def __init__(self, output_size, dropout=0.1, seq=400) -> None:
        super().__init__()
        self.seq = seq

        self.output_size = output_size

        self.bvp_encoder = SignalEncoder(self.output_size, dropout)
        self.eda_encoder = SignalEncoder(self.output_size, dropout)
        self.temp_encoder = SignalEncoder(self.output_size, dropout)
        self.hr_encoder = SignalEncoder(self.output_size, dropout)

    def forward(self, x):
        bvp = x[:, 0, :].reshape(-1, 1, self.seq)
        eda = x[:, 1, :].reshape(-1, 1, self.seq)
        temp = x[:, 2, :].reshape(-1, 1, self.seq)
        hr = x[:, 3, :].reshape(-1, 1, self.seq)

        bvp_encoder = self.bvp_encoder(bvp)
        eda_encoder = self.eda_encoder(eda)
        temp_encoder = self.temp_encoder(temp)
        hr_encoder = self.hr_encoder(hr)

        # output: [batch_size, feature, seq_len]
        outputs = [bvp_encoder, eda_encoder, temp_encoder, hr_encoder]

        encoder_outputs = torch.stack(outputs, 1)
        # output: [batch_size, feature, seq_len]
        # encoder_outputs = encoder_outputs.permute(0, 2, 1)
        return encoder_outputs


class SignalDecoder(nn.Module):
    def __init__(self, device, maskp) -> None:
        super().__init__()
        # self.maskp = maskp
        self.device = device
        self.decoder = TransformerDecoderLayer(
            d_model=4, nhead=4, dropout=0.1, batch_first=True)

    def forward(self, x, tgt):
        # mask = (torch.rand((tgt.shape[0], 4, 400), device=self.device)
        #         <= self.maskp).int()

        # tgt = tgt * mask
        tgt = tgt.permute(0, 2, 1)

        decoder_outputs = self.decoder(tgt, x)
        decoder_outputs = decoder_outputs.permute(0, 2, 1)

        return decoder_outputs


class ProjectionHead(nn.Module):
    def __init__(self, encoder_output, seq) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(encoder_output, 128)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.LeakyReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(128, 256)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.LeakyReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(256, seq)
        xavier_uniform_(self.hidden3.weight)
        # self.fcn = nn.Sequential(
        #     nn.Linear(encoder_output, 128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, seq)
        # )
        # kaiming_uniform_(self.fcn.weight, nonlinearity='relu')

    def forward(self, x):
        # input to first hidden layer
        x = self.hidden1(x)
        x = self.act1(x)
        # second hidden layer
        x = self.hidden2(x)
        x = self.act2(x)
        # output layer
        x = self.hidden3(x)
        return x


class MultiSignalRepresentation(nn.Module):

    def __init__(self, output_size, channel_maskp=0.5, pretrained=False, dropout=0.1, seq=400, signal_maskp=0.8, device=torch.device("cpu")):
        super().__init__()

        self.seq = seq
        self.output_size = output_size
        self.signal_maskp = signal_maskp
        self.channel_maskp = channel_maskp
        self.device = device
        self.pretrained = pretrained

        self.encoder = MultiSignalEncoder(
            output_size=self.output_size, seq=self.seq)

        self.output_layer = ProjectionHead(
            encoder_output=self.output_size, seq=self.seq)

        # self.fcn = nn.Sequential(
        #     nn.Linear(output_size, 128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 2)
        # )

    def waveform_masking(self, batch_size, channels, seq_len):
        noise_factor = 0.05
        synthetic_pattern = np.sin(
            np.arange(0, seq_len) * 2 * np.pi/(24*64))

        synthetic_pattern = synthetic_pattern / np.max(synthetic_pattern)
        masked_waveform = noise_factor * \
            np.random.normal(size=seq_len) + synthetic_pattern

        channels_mask = np.array([masked_waveform for i in range(channels)])
        batch_mask = np.array([channels_mask for i in range(batch_size)])

        return torch.from_numpy(batch_mask).to(device=self.device)

    def multivariate_masking(self, batch_size, channels, seq_len):
        multivariate_mask = [np.random.choice(
            [0, 1], size=channels, p=[self.channel_maskp, 1-self.channel_maskp]) for i in range(batch_size)]
        multivariate_mask = [mask.tolist() if np.sum(mask) != 0 else [
            1, 1, 1, 1] for mask in multivariate_mask]
        multivariate_mask = np.asarray([[[i] * seq_len for i in mask]
                                        for mask in multivariate_mask]).astype(int)
        multivariate_mask = torch.from_numpy(
            multivariate_mask).to(device=self.device).float()
        return multivariate_mask

    def time_masking(self, batch_size, channels, seq_len):
        return (torch.rand((batch_size, channels, seq_len), device=self.device)
                <= self.signal_maskp).int()

    def masking_generator(self, x):
        batch_size, channels, seq_len = x.shape[0], x.shape[1], x.shape[2]
        masked_waveform = self.waveform_masking(batch_size, channels, seq_len)
        masked_multivariate = self.multivariate_masking(
            batch_size, channels, seq_len)
        masked_time = self.time_masking(batch_size, channels, seq_len)

        x = ((x + masked_waveform) * masked_multivariate * masked_time).float()
        return x

    def forward(self, x):
        if not self.pretrained:
            x = self.masking_generator(x)

        encoder_outputs = self.encoder(x)
        output = self.output_layer(encoder_outputs)

        return output

        # encoder_outputs = self.encoder(x)
        # output = self.fcn(encoder_outputs)
        # return output


# class MultiSignalRepresentation(nn.Module):

#     def __init__(self, output_size, dropout=0.1, seq=400, maskp=0.8, device=torch.device("cpu")):
#         super().__init__()

#         self.seq = seq
#         self.output_size = output_size
#         self.maskp = maskp
#         self.device = device

#         self.encoder = MultiEncoder(output_size=self.output_size, seq=self.seq)

#         self.decoder = TransformerDecoderLayer(
#             d_model=4, nhead=4, dropout=0.1, batch_first=True)

#     def forward(self, x, tgt):
#         encoder_outputs = self.encoder(x)
#         mask = (torch.rand((tgt.shape[0], 4, 400), device=self.device)
#                 <= self.maskp).int()

#         tgt = tgt * mask
#         tgt = tgt.permute(0, 2, 1)

#         decoder_outputs = self.decoder(tgt, encoder_outputs)

#         decoder_outputs = decoder_outputs.permute(0, 2, 1)

#         return decoder_outputs
