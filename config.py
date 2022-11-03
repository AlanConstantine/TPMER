from torchmetrics.functional import auc, mean_squared_error
from torchmetrics import F1Score
from torchmetrics import AUC, MeanSquaredError
import torch


class Params(object):
    def __init__(self, model='CNNLSTM',
                 use_cuda=True,
                 debug=True,
                 lr=0.01,
                 epochs=100,
                 valid='loso',
                 target='valence_label',
                 batch_size=16,
                 out_channels=64,
                 hidden_size=256,
                 num_layers=6,
                 fcn_input=201728,
                 save_path='./output',
                 init=True
                 ):
        self.model = model
        self.use_cuda = use_cuda
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.batch_size = batch_size
        self.valid = valid
        self.target = target
        self.epochs = epochs
        self.lr = lr

        self.init = init

        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fcn_input = fcn_input

        self.debug = debug
        self.metrics_dict = {}
        if self.target in ['valence', 'arousal']:
            self.metrics_dict = {'mse': MeanSquaredError()}
        else:
            self.metrics_dict = {'f1': F1Score().to(self.device),
                                 #  'auc': AUC()
                                 }

        self.save_path = save_path
        self.k = None
        self.results = []
