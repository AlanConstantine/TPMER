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
from copy import deepcopy


class Params(object):

    def __init__(
            self,
            dataset='HKU',
            model='RP',
            use_cuda=True,
            debug=False,
            lr=0.000001,
            epochs=1000,
            valid='loso',
            target='valence',
            batch_size=32,
            dropout=0.1,
            init=True,
            show_wei=False,
            maskp=0.5):

        self.model = model

        self.show_wei = show_wei
        self.use_cuda = use_cuda
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.metrics = {'mse': MeanSquaredError().to(self.device)}
        self.metrics_val = deepcopy(self.metrics)
        self.batch_size = batch_size
        self.valid = valid
        self.debug = debug
        self.target = target
        self.epochs = epochs
        self.maskp = maskp
        if self.debug:
            self.epochs = 5
            self.batch_size = 8
        self.lr = lr

        self.init = init

        self.dropout = dropout

        self.save_path = './output/rep_{}_{}_{}_maskp{}'.format(
            lr, batch_size, int(time.time()), maskp)
        self.k = None
        self.results = {}
        if not self.debug:
            self.create_log_folder()
        self.checkpoint = os.path.join(self.save_path, '{}_{}_maskp{}_checkpoint.pt'.format(
            lr, batch_size, maskp))

    def create_log_folder(self):
        if not os.path.exists(r'./output'):
            os.makedirs(r'./output')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def save_results(self, results):
        with open(os.path.join(self.save_path, 'results.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
