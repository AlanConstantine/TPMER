# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2023-01-16 15:30:02

# from tools import *
# from CONSTANT import *

from param import Params
from tools import *

from SigRepre import MultiSignalRepresentation


from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import math
import datetime
import os
import sys
import time
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')


torch.manual_seed(3407)


def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_normal_(m.weight)


def train():
    pass


def eval():
    pass


def run():
    pass


def main():
    args = Params()
    dataprepare = DataPrepare(args)
    train_dataloader, test_dataloader = dataprepare.get_data()

    model = MultiSignalRepresentation(output_size=40)


if __name__ == '__main__':
    main()
