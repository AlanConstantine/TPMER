from torchmetrics.functional import auc, mean_squared_error
from torchmetrics import F1Score


class Params(object):
    def __init__(self, model='CNNLSTM',
                 use_cuda=False,
                 debug=True,
                 lr=0.01,
                 epochs=100,
                 valid='loso',
                 target='valence_label',
                 batch_size=8,
                 out_channels=64,
                 hidden_size=256,
                 num_layers=3,
                 fcn_input=201728
                 ):
        self.model = model
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.valid = valid
        self.target = target
        self.epochs = epochs
        self.lr = lr

        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fcn_input = fcn_input

        self.debug = debug
        self.metrics_dict = {}
        if self.target in ['valence', 'arousal']:
            self.metrics_dict = {'mse': mean_squared_error}
        else:
            self.metrics_dict = {'f1': F1Score, 'auc': auc}
        self.k = None
        self.results = []

    def print_args(self):
        print()
