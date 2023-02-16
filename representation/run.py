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


def train(args, model, optimizer, scheduler, loss_fn, train_dataloader):
    model.train()

    total_loss = 0

    loop = tqdm(enumerate(train_dataloader),
                total=len(train_dataloader),
                file=sys.stdout)

    for i, batch in loop:
        preds = model(batch)

        loss = loss_fn(preds, batch)

        loss.backforward()
        optimizer.step()

        optimizer.zero_grad()

        total_loss += loss.item()

    # return loss.item()


def eval(args, model, optimizer, scheduler, loss_fn, test_dataloader):
    model.eval()
    preds = model()
    # loss = loss_fn(preds, batch)


def run(args, model, optimizer, scheduler, loss_fn, train_dataloader,
        test_dataloader):
    history = {}

    if args.init:
        model.apply(init_xavier)

    model = train(args, model, optimizer, scheduler, loss_fn, train_dataloader)

    return history


def main():
    args = Params()
    dataprepare = DataPrepare(args)
    train_dataloader, test_dataloader = dataprepare.get_data()

    model = MultiSignalRepresentation(output_size=40)
    model = model.to(args.device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=5,
                                  verbose=True,
                                  threshold_mode='rel',
                                  cooldown=0,
                                  min_lr=0,
                                  eps=1e-08)

    run(args, model, optimizer, scheduler, loss_fn, train_dataloader,
        test_dataloader)


if __name__ == '__main__':
    main()
