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
            nn.Conv1d(16, 4, 3),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )

        self.lstm1 = nn.LSTM(input_size=4,
                             hidden_size=8,
                             num_layers=2, batch_first=True,
                             bidirectional=True
                             )

        self.lstm2 = nn.LSTM(input_size=16,
                             hidden_size=16,
                             num_layers=2, batch_first=True,
                             bidirectional=True
                             )

        self.fcn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(24384, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
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
