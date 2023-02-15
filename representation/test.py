import torch
from torch import nn
import pandas as pd
from tools import *
from param import Params
from SigRepre import MultiSignalRepresentation

args = Params(use_cuda=True)
dataprepare = DataPrepare(args)

model = MultiSignalRepresentation(output_size=40)
model.to(args.device)
xtest = dataprepare.X_test[:10]
print(model(xtest, xtest).shape)
