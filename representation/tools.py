# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2023-01-25 23:35:09

import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import datetime
from torch import nn
from torch.utils.data import (TensorDataset, DataLoader, SequentialSampler,
                              RandomSampler)
from sklearn.model_selection import train_test_split

torch.manual_seed(3407)


def concat_signals(df):
    bvp_cols = [
        fea for fea in df.columns.values if fea.split('_')[0] in ['BVP']
    ]
    eda_cols = [
        fea for fea in df.columns.values if fea.split('_')[0] in ['EDA']
    ]
    temp_cols = [
        fea for fea in df.columns.values if fea.split('_')[0] in ['TEMP']
    ]
    hr_cols = [fea for fea in df.columns.values if fea.split('_')[0] in ['HR']]

    signals = []
    for bvp, eda, temp, hr in zip(df[bvp_cols].values, df[eda_cols].values,
                                  df[temp_cols].values, df[hr_cols].values):
        signals.append([bvp, eda, temp, hr])

    return np.array(signals)


class DataPrepare(object):

    def __init__(
            self,
            args,
            datapath=r'../processed_signal/all_400_4s_step_2s.pkl') -> None:
        self.args = args
        if self.args.debug:
            datapath = r'../processed_signal/all_768_12s_step_2s_sampled.pkl'
        self.df = pd.read_pickle(datapath)
        self.drop_columns()
        self.randomization()
        print('Data size:', self.df.shape)

        X = concat_signals(self.df)

        X_train, X_test = train_test_split(X,
                                           test_size=0.2,
                                           random_state=3407,
                                           shuffle=True)

        if self.args.debug:
            X_train, X_test = X_train[:1000], X_test[:100]

        X_train = torch.from_numpy(X_train).to(torch.float32)
        X_test = torch.from_numpy(X_test).to(torch.float32)

        self.X_train, self.X_test = X_train.to(self.args.device), X_test.to(
            self.args.device)

        self.batch_size = self.args.batch_size

        print('Splited size:', X_train.shape, X_test.shape)

    def randomization(self):
        print('Data shuffled')
        self.df = self.df.sample(frac=1)

    def drop_columns(self):
        self.df = self.df.loc[:, ~self.df.columns.
                              isin(['participant_id', 'source'])]

    def get_data(self):
        train_data = TensorDataset(self.X_train)
        test_data = TensorDataset(self.X_test)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=self.batch_size,
                                      drop_last=False)

        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data,
                                     sampler=test_sampler,
                                     batch_size=self.batch_size,
                                     drop_last=False)

        return train_dataloader, test_dataloader


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "======" * 6 + "[%s]" % nowtime + "======" * 6)
    print(str(info) + "\n")


class StepRunner:

    def __init__(self,
                 model,
                 loss_fn,
                 metrics=None,
                 stage='train',
                 optimizer=None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.stage = stage
        self.optimizer = optimizer
        self.sig = nn.Sigmoid()

    def step(self, features, labels):
        preds = self.model(features)
        if self.optimizer and self.stage == 'train':
            self.optimizer.zero_grad()
        # get loss
        loss = self.loss_fn(preds, labels)
        # backward
        if self.stage == 'train':
            loss.backward()
            self.optimizer.step()

        # get metrics
        step_metrics = {}
        if self.metrics:
            for metric_name, metric_fn in self.metrics.items():
                if metric_name == 'f1':
                    step_metrics[self.stage + "_" + metric_name] = metric_fn(
                        torch.round(self.sig(preds)).long(), labels).item()
                elif metric_name == 'acc':
                    step_metrics[self.stage + "_" + metric_name] = metric_fn(
                        torch.round(preds), labels).item()
                elif metric_name == 'auc':
                    step_metrics[self.stage + "_" + metric_name] = metric_fn(
                        torch.round(self.sig(preds)).long(), labels).item()
                elif metric_name in ['mse', 'mae']:
                    step_metrics[self.stage + "_" + metric_name] = metric_fn(
                        preds, labels).item()
        return loss.item(), step_metrics

    def train_step(self, features, labels):
        self.model.train()
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.model.eval()
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == 'train':
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:

    def __init__(self, steprunner, metrics=None) -> None:
        self.steprunner = steprunner
        self.stage = self.steprunner.stage
        self.metrics = metrics

    def __call__(self, dataloader):
        total_loss, step = 0, 0

        epoch_metrics = {}
        if self.metrics:
            epoch_metrics = {
                self.stage + '_' + name: 0
                for name in self.metrics.keys()
            }

        loop = tqdm(enumerate(dataloader),
                    total=len(dataloader),
                    file=sys.stdout)

        for i, batch in loop:
            if len(batch) == 1:
                loss, step_metrics = self.steprunner(batch[0], batch[0])
            else:
                loss, step_metrics = self.steprunner(*batch)
            step_log = {}
            if len(step_metrics) != 0:
                step_log = dict({self.stage + "_loss": loss}, **step_metrics)
                for metric_name, scores in step_metrics.items():
                    epoch_metrics[metric_name] += scores
            total_loss += loss
            step += 1

            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                if len(step_metrics) != 0:
                    for metric_name, scores in step_metrics.items():
                        epoch_metrics[metric_name] = round(
                            epoch_metrics[metric_name] / step, 4)
                epoch_log = dict({self.stage + "_loss": epoch_loss},
                                 **epoch_metrics)
                loop.set_postfix(**epoch_log)

        epoch_log.update(epoch_metrics)

        return epoch_log
