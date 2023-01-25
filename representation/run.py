# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2023-01-16 15:30:02

# from tools import *
# from CONSTANT import *

from torch.utils.data import (
    TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler)
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time
import datetime
import os
import sys
import time
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')


torch.manual_seed(3480)


def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_normal_(m.weight)


class DataPrepare(object):
    def __init__(self, args, datapath=r'../processed_signal/all_400_4s_step_2s.pkl') -> None:
        self.args = args
        self.data = pd.read_pickle(datapath)
        self.drop_columns()
        self.randomization()
        print('Data size:', self.data.shape)

        X_train, X_test = train_test_split(
            self.data, test_size=0.2, random_state=3131, shuffle=True)

        if self.args.debug:
            X_train, X_test = X_train[:1000], X_test[:100]

        X_train = torch.from_numpy(X_train).to(torch.float32)
        X_test = torch.from_numpy(X_test).to(torch.float32)

        self.X_train, self.X_test = X_train.to(
            self.args.device), X_test.to(self.args.device)

        self.batch_size = self.args.batch_size

        print('Splited size:', X_train.shape, X_test.shape)

    def randomization(self):
        self.data = self.data.sample(frac=1)

    def drop_columns(self):
        self.data = self.data.loc[:, ~self.data.columns.isin(
            ['participant_id', 'source'])]


def train():
    pass


def eval():
    pass


def run():
    pass


def main():
    dataprepare = DataPrepare()


if __name__ == '__main__':
    main()
