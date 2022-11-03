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

        self.fcn = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(args.fcn_input, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(512, 2)

        self.regresser = nn.Linear(512, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        input [batch_size, channels, seq_len]
        """
        x = self.cnns(x)  # output [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)  # output [batch_size, seq_len, Hin]
        x = self.relu(x)
        x, _ = self.lstm2(x)  # output [batch_size, seq_len, Hin]
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        print(x.shape)
        x = self.fcn(x)
        if self.args.target in ['valence', 'arousal']:
            output = self.regresser(x)
            return output
        else:
            output = torch.sigmoid(self.classifier(x))
            return output
