import torch
from torch import nn
import torch.nn.functional as F


class CNNBiLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
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

        self.lstm1 = nn.LSTM(input_size=args.out_channels,
                             hidden_size=64,
                             num_layers=args.nlayers, batch_first=True,
                             bidirectional=True
                             )

        self.lstm2 = nn.LSTM(input_size=64 * 2,
                             hidden_size=args.hidden_size,
                             num_layers=args.nlayers, batch_first=True,
                             bidirectional=True
                             )

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
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)  # output [batch_size, seq_len, Hin]
        x = F.relu(x)
        x, _ = self.lstm2(x)  # output [batch_size, seq_len, Hin]
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        output = self.fcn(x)
        return output
