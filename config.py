# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2022-11-16 23:41:33

from torchmetrics import F1Score, Accuracy
from torchmetrics import MeanSquaredError
import torch
from torch import nn
import os
import pickle


class Params(object):

    def __init__(
            self,
            dataset='HKU',
            model='SG',
            use_cuda=True,
            debug=False,
            lr=0.0001,
            epochs=200,
            valid='loso',
            target='valence',
            batch_size=256,
            dropout=0.2,
            out_channels=32,
            hidden_size=64,  # lstm hidden_size
            nlayers=2,  # transformer or lstm layer num
            nhead=4,  # transformer head num
            #  fcn_input=50432,  # LSTM fcn num
        fcn_input=12608,  # Transformer fcn num
            init=True,
            show_wei=False,
            pretrain=True):

        self.data = r'./processed_signal/HKU956/400_4s_step_2s.pkl'
        self.spliter = r'./processed_signal/HKU956/400_4s_step_2s_spliter10.pkl'
        if dataset == 'KEC':
            self.data = r'./processed_signal/KEmoCon/KEC_400.pkl'
            self.spliter = r'./processed_signal/KEmoCon/KEC_400_spliter10.pkl'
        if dataset == 'WES':
            self.data = r'./processed_signal/WESAD/400_4s_step_2s.pkl'
            self.spliter = r'./processed_signal/WESAD/400_4s_step_2s_spliter10.pkl'
        self.model = model

        self.pretrain = pretrain
        self.pretrain_model = ''
        if self.pretrain and self.model == 'CT':
            self.pretrain_model = r'./output/HKU956/valence_CTransformer_loso_0.0001_256_32/fold2_checkpoint.pt'
        elif self.pretrain and self.model == 'SG':
            self.pretrain_model = r'./output/False_WES_valence_SG_loso_0.0001_512_32/fold4_checkpoint.pt'

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

        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.fcn_input = fcn_input

        self.nlayers = nlayers
        self.nhead = nhead

        self.metrics_dict = {}
        if self.target in ['valence', 'arousal']:
            self.metrics_dict = {'mse': MeanSquaredError().to(self.device)}
        else:
            self.metrics_dict = {
                'f1': F1Score().to(self.device),
                'acc': Accuracy().to(self.device),
                # 'auc': AUC().to(self.device)
            }

        self.save_path = './output/{}_{}_{}_{}_{}_{}_{}_{}'.format(
            self.pretrain, dataset, target, model, valid, lr, batch_size,
            out_channels)
        self.k = None
        self.results = {}
        if not self.debug:
            self.create_log_folder()

        self.fcn = nn.Sequential(nn.Dropout(p=0.2),
                                 nn.Linear(self.fcn_input, 128), nn.ReLU(),
                                 nn.Dropout(p=0.2), nn.Linear(128, 32),
                                 nn.ReLU(), nn.Linear(32, 1))

    def create_log_folder(self):
        if not os.path.exists(r'./output'):
            os.makedirs(r'./output')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def save_results(self, results):
        with open(os.path.join(self.save_path, 'results.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
