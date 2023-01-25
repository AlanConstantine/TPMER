# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2023-01-25 23:35:09


import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    TensorDataset, DataLoader, SequentialSampler, RandomSampler)
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
    def __init__(self, args, datapath=r'../processed_signal/all_400_4s_step_2s.pkl') -> None:
        self.args = args
        self.df = pd.read_pickle(datapath)
        self.drop_columns()
        self.randomization()
        print('Data size:', self.df.shape)

        X = concat_signals(self.df)

        X_train, X_test = train_test_split(
            X, test_size=0.2, random_state=3407, shuffle=True)

        if self.args.debug:
            X_train, X_test = X_train[:1000], X_test[:100]

        X_train = torch.from_numpy(X_train).to(torch.float32)
        X_test = torch.from_numpy(X_test).to(torch.float32)

        self.X_train, self.X_test = X_train.to(
            self.args.device), X_test.to(self.args.device)

        self.batch_size = self.args.batch_size

        print('Splited size:', X_train.shape, X_test.shape)

    def randomization(self):
        print('Data shuffled')
        self.df = self.df.sample(frac=1)

    def drop_columns(self):
        self.df = self.df.loc[:, ~self.df.columns.isin(
            ['participant_id', 'source'])]

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
