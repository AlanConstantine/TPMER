# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2023-01-16 15:30:02

from tools import *
from CONSTANT import *

from torch.utils.data import (
    TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler)
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
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


def train():
    pass


def eval():
    pass


def run():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
