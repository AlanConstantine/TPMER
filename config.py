from torchmetrics.functional import auc, mean_squared_error
from torchmetrics import F1Score, Accuracy
from torchmetrics import AUC, MeanSquaredError
import torch
import os
import pickle


class Params(object):
    def __init__(self,
                 model='CTransformer',
                 use_cuda=True,
                 debug=False,
                 lr=0.0001,
                 epochs=200,
                 valid='loso',
                 target='arousal',
                 batch_size=256,
                 out_channels=32,
                 hidden_size=64,  # lstm hidden_size
                 nlayers=2,  # transformer or lstm layer num
                 nhead=4,  # transformer head num
                 #  fcn_input=50432,  # LSTM fcn num
                 fcn_input=25216,  # Transformer fcn num
                 init=True,
                 show_wei=False
                 ):
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

        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.fcn_input = fcn_input

        self.nlayers = nlayers
        self.nhead = nhead

        self.metrics_dict = {}
        if self.target in ['valence', 'arousal']:
            self.metrics_dict = {'mse': MeanSquaredError().to(self.device)}
        else:
            self.metrics_dict = {'f1': F1Score().to(self.device),
                                 'acc':  Accuracy().to(self.device),
                                 # 'auc': AUC().to(self.device)
                                 }

        self.save_path = './output/{}_{}_{}_{}_{}_{}'.format(
            target, model, valid, lr, batch_size, out_channels)
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
