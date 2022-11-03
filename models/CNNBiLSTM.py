import torch
from torch import nn


class CNNBiLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cnns = nn.Sequential(
            nn.Conv1d(4, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, args.out_channels, 3),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.relu = nn.ReLU()

        self.output = nn.Linear(512, 2)
        if self.args.target in ['valence', 'arousal']:
            self.output = nn.Linear(512, 1)

        # self.classifier = nn.Linear(512, 2)

        # self.regresser = nn.Linear(512, 1)

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
        x = self.fcn(x)
        if self.args.target in ['valence', 'arousal']:
            output = self.output(x)
            return output
        else:
            output = torch.sigmoid(self.output(x))
            return output


# pytorch计算图、梯度相关操作、固定参数训练以及训练过程中grad为Nonetype的原因https://zhuanlan.zhihu.com/p/438630330

# class NN(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.nn = nn.Sequential(nn.Linear(400 * 4, 1024),
#                                 nn.ReLU(),
#                                 nn.Linear(1024, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(512, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(512, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(512, 256),
#                                 nn.ReLU(),
#                                 )

#         self.output = nn.Linear(256, 2)
#         if self.args.target in ['valence', 'arousal']:
#             self.output = nn.Linear(256, 1)

#     def forwar(self, x):
#         x = x.flatten(start_dim=1)
#         x = self.nn(x)
#         if self.args.target in ['valence', 'arousal']:
#             output = self.output(x)
#             return output
#         else:
#             output = torch.sigmoid(self.output(x))
#             return output
