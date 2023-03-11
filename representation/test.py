import torch
from torch import nn
import pandas as pd
from tools import *
from param import Params
from SigRepre import MultiSignalRepresentation

args = Params(use_cuda=True, )
dataprepare = DataPrepare(args)

model = MultiSignalRepresentation(seq=768,
                                  output_size=40, pretrained=False, device=args.device)
model.to(args.device)
xtest = dataprepare.X_test[:2]
print(model(xtest).shape)
