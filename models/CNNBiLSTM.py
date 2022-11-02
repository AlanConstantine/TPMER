import torch
from torch import nn


class CNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnns = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.Sequential()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )

    def forward():
        """
        input shape [batch_size, channels, seq_len]
        """
        pass
