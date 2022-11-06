from torchmetrics.functional import auc, mean_squared_error
from torchmetrics import F1Score
from torchmetrics import AUC, MeanSquaredError
import torch


class Params(object):
    def __init__(self, model='CLSTM',
                 use_cuda=True,
                 debug=False,
                 lr=0.001,
                 epochs=5,
                 valid='loso',
                 target='valence',
                 batch_size=128,
                 out_channels=32,
                 hidden_size=64,
                 num_layers=1,
                 fcn_input=50432,
                 save_path='./output',
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
            self.metrics_dict = {'mse': MeanSquaredError().to(self.device)}
        else:
            self.metrics_dict = {'f1': F1Score().to(self.device),
                                 #  'auc': AUC().to(self.device)
                                 }

        self.save_path = save_path
        self.k = None
        self.results = []
