# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2023-01-25 23:25:54

from torchmetrics import F1Score, Accuracy
from torchmetrics import MeanSquaredError
import torch
from torch import nn
import os
import pickle
import time


class Params(object):

    def __init__(
            self,
            dataset='HKU',
            model='RP',
            use_cuda=True,
            debug=True,
            lr=0.0001,
            epochs=200,
            valid='loso',
            target='valence',
            batch_size=256,
            dropout=0.2,
            init=True,
            show_wei=False,):

        self.model = model

        self.show_wei = show_wei
        self.use_cuda = use_cuda
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.batch_size = batch_size
        self.valid = valid
        self.debug = debug
        self.target = target
        self.epochs = epochs
        if self.debug:
            self.epochs = 5
        self.lr = lr

        self.init = init

        self.dropout = dropout

        self.save_path = './output/{}_{}_{}_{}_{}_{}_{}'.format(
            dataset, target, model, valid, lr, batch_size, int(time.time()))
        self.k = None
        self.results = {}
        if not self.debug:
            self.create_log_folder()

    def create_log_folder(self):
        if not os.path.exists(r'./output'):
            os.makedirs(r'./output')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def save_results(self, results):
        with open(os.path.join(self.save_path, 'results.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
