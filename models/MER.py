import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from representation.SigRepre import MultiSignalRepresentation


class MERClassifer(nn.Module):
    def __init__(self, args, n_class, input_size=160) -> None:
        super().__init__()
        # define model elements

        # input to first hidden layer
        self.hidden1 = nn.Linear(input_size, 128)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(128, 32)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(32, n_class)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = X.flatten(start_dim=1)
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X
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
        self.hidden1 = nn.Linear(input_size, 128)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(128, 32)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(32, 1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = X.flatten(start_dim=1)

        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        return X
