# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2022-11-16 23:41:33

from torchmetrics import F1Score, Accuracy
from torchmetrics import MeanSquaredError
import torch
from torch import nn
import os
import pickle
import time

torch.cuda.set_device(0)


class Params(object):

    def __init__(
            self,
            dataset='HKU',
            model='SG',
            use_cuda=True,
            debug=False,

            lr=0.001,
            # lr=0.0001,
            epochs=200,
            valid='cv',
            target='valence_label',
            batch_size=8,
            dropout=0.2,
            out_channels=32,
            hidden_size=64,  # lstm hidden_size
            nlayers=2,  # transformer or lstm layer num
            nhead=4,  # transformer head num
            #  fcn_input=50432,  # LSTM fcn num
        fcn_input=12608,  # Transformer fcn num
            init=True,
            show_wei=False,
            data=r'./processed_signal/HKU956/1540_24s_step_12s.pkl',
            spliter=r'./processed_signal/HKU956/1540_24s_step_12s_spliter5.pkl',
            # pretrain=False
            pretrain=r'./representation/output/rep_1e-06_32_1678067974_maskp0.8/1e-06_32_maskp0.8_checkpoint.pt'
            # pretrain=r'./representation/output/1e-06_32_maskp0.8_checkpoint.pt'
            # pretrain=r'./representation/output/0.0001_256_1677293682_maskp0.8/0.0001_256_maskp0.8_checkpoint.pt'
            # pretrain=r'./representation/output/rep_0.0001_128_1677704406_maskp0.8/0.0001_128_maskp0.8_checkpoint.pt'
    ):
        self.data = data

        self.spliter = spliter

        data_name = os.path.split(data)[-1].replace('.pkl', '')
        self.input_size = int(self.data.split('_')[-4].split('/')[-1]) - 4
        # self.data = r'./processed_signal/HKU956/400_4s_step_2s.pkl'
        # self.data = r'./processed_signal/HKU956/last15_400_4s_step_2s.pkl'

        # self.spliter = r'./processed_signal/HKU956/last15_400_4s_step_2s_spliter10.pkl'

        if dataset == 'KEC':
            self.data = r'./processed_signal/KEmoCon/KEC_400.pkl'
            self.spliter = r'./processed_signal/KEmoCon/KEC_400_spliter10.pkl'
        if dataset == 'WES':
            self.data = r'./processed_signal/WESAD/400_4s_step_2s.pkl'
            self.spliter = r'./processed_signal/WESAD/400_4s_step_2s_spliter10.pkl'
        self.model = model

        self.pretrain = pretrain
        self.pretrain_model = ''
        # if self.pretrain and self.model == 'CT':
        #     self.pretrain_model = r'./output/HKU956/valence_CTransformer_loso_0.0001_256_32/fold2_checkpoint.pt'
        # elif self.pretrain and self.model == 'SG':
        #     self.pretrain_model = r'./output/False_WES_valence_SG_loso_0.0001_512_32/fold4_checkpoint.pt'

        self.show_wei = show_wei
        self.use_cuda = use_cuda
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.batch_size = batch_size
        self.valid = valid
        self.debug = debug
        self.target = target
        self.epochs = epochs
        if self.debug:
            self.epochs = 20
            self.batch_size = 2
        self.lr = lr

        self.init = init

        self.dropout = dropout

        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.fcn_input = fcn_input

        self.nlayers = nlayers
        self.nhead = nhead

        self.metrics_dict = {}
        if self.target in ['valence_rating', 'arousal_rating']:
            self.metrics_dict = {'mse': MeanSquaredError().to(self.device)}
        else:
            self.metrics_dict = {
                'f1': F1Score(task='binary', average='macro', num_classes=2).to(self.device),
                'acc': Accuracy(task='binary', num_classes=2).to(self.device),
                # 'auc': AUC().to(self.device)
            }

        self.save_path = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
            dataset, target, model, valid, lr, batch_size, data_name, int(time.time()))
        if 'output' not in self.save_path:
            self.save_path = os.path.join(r'./output', self.save_path)
        self.k = None
        # self.results = {'valid_clf_report': []}
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
