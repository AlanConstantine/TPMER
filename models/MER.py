import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


class MERClassifer(nn.Module):
    def __init__(self, args, n_class, input_size=160) -> None:
        super().__init__()
        # define model elements

        # input to first hidden layer
        self.hidden1 = nn.Linear(input_size, 256)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(256, 64)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(64, n_class)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Softmax(dim=1)

    # forward propagate input
    def forward(self, x):
        # input to first hidden layer
        x = x.flatten(start_dim=1)
        x = self.hidden1(x)
        x = self.act1(x)
        # second hidden layer
        x = self.hidden2(x)
        x = self.act2(x)
        # output layer
        x = self.hidden3(x)
        x = self.act3(x)
        return x
        # self.pretrain_model = MultiSignalRepresentation(
        #     output_size=40, device=args.device)
        # self.pretrain_model.load_state_dict(
        #     torch.load(r'./representation/mask08ep13.pt'))
        # self.encoder = self.pretrain_mode.encoder
        # self.fcn = nn.Sequential(
        #     nn.Linear(output_size, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, n_class)
        # )

    # def forward(self, x, y):
    #     x = x.flatten(start_dim=1)
    #     return self.fcn(x)


class MERRegressor(nn.Module):
    def __init__(self, input_size=160) -> None:
        super().__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(input_size, 256)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(256, 128)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(32, 1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, x):
        # input to first hidden layer
        x = x.flatten(start_dim=1)

        x = self.hidden1(x)
        x = self.act1(x)
        # second hidden layer
        x = self.hidden2(x)
        x = self.act2(x)
        # output layer
        x = self.hidden3(x)
        return x


class SignalSample(nn.Module):
    def __init__(self, input_size=768, output_size=768) -> None:
        super().__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(input_size, 512)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(512, output_size)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.output_layer = nn.Linear(32, 1)
        xavier_uniform_(self.output_layer.weight)

    # forward propagate input
    def forward(self, x):
        # input to first hidden layer
        x = self.hidden1(x)
        x = self.act1(x)
        # second hidden layer
        x = self.hidden2(x)
        x = self.act2(x)
        # output layer
        x = self.output_layer(x)
        return x
