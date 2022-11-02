# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2022-11-02 17:26:08


from argparse import ArgumentParser
from torchmetrics.functional import auc, mean_squared_error
from torchmetrics import F1Score
from tools import *
from CONSTANT import *
from models import CNNBiLSTM, Transformer
from config import Params
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


torch.manual_seed(31)


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s" % nowtime)
    print(str(info)+"\n")


class StepRunner:
    def __init__(self, net, loss_fn,
                 stage="train", metrics_dict=None,
                 optimizer=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer = optimizer

    def step(self, features, labels):
        # loss
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {self.stage+"_"+name: metric_fn(preds, labels).item()
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics

    def train_step(self, features, labels):
        self.net.train()  # 训练模式, dropout层发生作用
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.net.eval()  # 预测模式, dropout层不发生作用
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(
            dataloader), file=sys.stdout)
        for i, batch in loop:
            loss, step_metrics = self.steprunner(*batch)
            step_log = dict({self.stage+"_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if i != len(dataloader)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss/step
                epoch_metrics = {self.stage+"_"+name: metric_fn.compute().item()
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict(
                    {self.stage+"_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(net, optimizer, loss_fn, metrics_dict,
                train_data, val_data=None,
                epochs=10, ckpt_path='checkpoint.pt',
                patience=5, monitor="val_loss", mode="min"):

    history = {}

    for epoch in range(1, epochs+1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(net=net, stage="train",
                                       loss_fn=loss_fn, metrics_dict=deepcopy(
                                           metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(net=net, stage="val",
                                         loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(
            arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores)-1:
            torch.save(net.state_dict(), ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                              arr_scores[best_score_idx]))
        if len(arr_scores)-best_score_idx > patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor, patience))
            break
        net.load_state_dict(torch.load(ckpt_path))

    return pd.DataFrame(history)


def run(train_dataloader, test_dataloader, args):
    model = None
    if args.model == 'CNNLSTM':
        model = CNNBiLSTM.CNNBiLSTM()
    else:
        pass
    loss_fn = nn.BCEWithLogitsLoss()
    mode = 'max'
    if args.target in ['valence', 'arousal']:
        loss_fn = nn.MSELoss()
        mode = "min"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics_dict = args.metrics_dict

    history = train_model(model,
                          optimizer,
                          loss_fn,
                          metrics_dict,
                          train_data=train_dataloader,
                          val_data=test_dataloader,
                          epochs=args.epochs,
                          patience=5,
                          monitor="val_{}".format(
                              list(metrics_dict.keys())[0]),
                          mode=mode)


def main():
    args = Params(debug=True)
    spliter = load_model(
        r'./processed_signal/HKU956/400_4s_step_2s_spliter.pkl')
    data = pd.read_pickle(r'./processed_signal/HKU956/400_4s_step_2s.pkl')
    print(args)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    for k in spliter[args.valid]:
        args.k = k
        print('[Fold {}]'.format(k), '='*31)
        train_index = k['train_index']
        test_index = k['test_index']
        dataprepare = DataPrepare(
            target='valence', data=data, train_index=train_index, test_index=test_index, device=device)
        train_dataloader, test_dataloader = dataprepare.get_data()
        run(train_dataloader, test_dataloader, args)
        if args.debug:
            break


if __name__ == '__main__':
    main()
