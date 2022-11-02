import torch
from torch import nn


class CNNBiLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cnns = nn.Sequential(
            nn.Conv1d(4, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, args.out_channels, 3),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.lstm1 = nn.LSTM(input_size=args.out_channels, 
                    hidden_size=512,
                    num_layers=args.num_layers, batch_first=True,
                    bidirectional=True
                    )

        self.lstm2 = nn.LSTM(input_size=512 * 2, 
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers, batch_first=True,
                    bidirectional=True
                    )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(args.hidden_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        input [batch_size, channels, seq_len]
        """
        x = self.cnns(x) # output [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)
        x, _ = lstms1(x) # output [batch_size, seq_len, Hin]
        x = self.relu(x)
        x, _ = lstms2(x) # output [batch_size, seq_len, Hin]
        x = self.relu(x)
        output = classifier(x)
        return output
