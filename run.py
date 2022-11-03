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
# warnings.filterwarnings('ignore')


torch.manual_seed(31)


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s" % nowtime)
    print(str(info)+"\n")


def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_normal_(m.weight)


class StepRunner:
    def __init__(self, net, loss_fn, args,
                 stage="train", metrics_dict=None,
                 optimizer=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer = optimizer
        self.args = args
        self.results = None

    def step(self, features, labels):
        # loss
        preds = self.net(features)

        if self.optimizer is not None and self.stage == "train":
            self.optimizer.zero_grad()

        loss = None
        if self.args.target in ['valence', 'arousal']:
            loss = self.loss_fn(preds, labels)
        else:
            loss = self.loss_fn(torch.argmax(
                preds, dim=1).reshape(-1, 1).to(torch.float32), labels.to(torch.float32))
        loss.requires_grad_(True)

        if self.stage == "train":
            loss.backward()
            self.optimizer.step()

        # metrics
        step_metrics = {}
        for name, metric_fn in self.metrics_dict.items():
            if self.args.target in ['valence', 'arousal']:
                step_metrics[self.stage+"_" +
                             name] = metric_fn(preds, labels).item()
            else:
                if name == 'f1':
                    step_metrics[self.stage+"_" +
                                 name] = metric_fn(torch.argmax(preds, dim=1), labels).item()
                elif name == 'auc':
                    step_metrics[self.stage+"_" +
                                 name] = metric_fn(preds[:, 1], labels).item()
                else:
                    pass
        self.results = step_metrics
        return loss.item(), step_metrics

    def train_step(self, features, labels):
        self.net.train()
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.net.eval()
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

                epoch_metrics = self.steprunner.results

                epoch_log = dict(
                    {self.stage+"_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

        return epoch_log


def train_model(args, net, optimizer, loss_fn, metrics_dict,
                train_data, val_data=None,
                epochs=10, ckpt_path='checkpoint.pt',
                patience=5, monitor="val_loss", mode="min"):

    history = {}

    if args.init:
        net.apply(init_xavier)

    for epoch in range(1, epochs+1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(net=net, stage="train",
                                       loss_fn=loss_fn, args=args, metrics_dict=deepcopy(
                                           metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(args=args, net=net, stage="val",
                                         loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        for name, parms in net.named_parameters():
            # print('-->name:', name)
            # print('-->grad_requirs:', parms.requires_grad)
            # print('--weight', torch.mean(parms.data))
            # print('-->grad_value:', torch.mean(parms.grad))
            if name == 'cnns.0.weight':
                print(name, torch.mean(parms.data))

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
    if args.model == 'CLSTM':
        model_ = CNNBiLSTM.CNNBiLSTM(args)
    else:
        pass
    model = model_.to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    mode = 'max'
    if args.target in ['valence', 'arousal']:
        loss_fn = nn.MSELoss()
        mode = "min"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics_dict = args.metrics_dict

    history = train_model(args, model,
                          optimizer,
                          loss_fn,
                          metrics_dict,
                          train_data=train_dataloader,
                          val_data=test_dataloader,
                          epochs=args.epochs,
                          patience=5,
                          monitor="val_{}".format(
                              list(metrics_dict.keys())[0]),
                          mode=mode, ckpt_path=os.path.join(
                              args.save_path, 'checkpoint.pt')
                          )


def main():
    args = Params(debug=True)
    spliter = load_model(
        r'./processed_signal/HKU956/400_4s_step_2s_spliter.pkl')
    data = pd.read_pickle(r'./processed_signal/HKU956/400_4s_step_2s.pkl')

    print('\n'.join("%s: %s" % item for item in vars(args).items()))

    for i, k in enumerate(spliter[args.valid]):
        args.k = i
        print('[Fold {}]'.format(i), '='*31)
        train_index = k['train_index']
        test_index = k['test_index']
        dataprepare = DataPrepare(args,
                                  target='valence', data=data, train_index=train_index, test_index=test_index, device=args.device, batch_size=args.batch_size)
        train_dataloader, test_dataloader = dataprepare.get_data()
        run(train_dataloader, test_dataloader, args)
        if args.debug:
            break


if __name__ == '__main__':
    main()
