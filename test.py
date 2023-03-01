import torch
from torch import nn
import pandas as pd
from tools import *
from config import Params
from representation.SigRepre import MultiSignalRepresentation
from models import MER

args = Params(use_cuda=True, )
dataprepare = DataPrepare(args, target='valence_label', data)

model = MultiSignalRepresentation(
    output_size=40, pretrained=False, device=args.device)
model.to(args.device)
model.output_layer = MER.MERClassifer(args, 2)
xtest = dataprepare.X_test[:2]
print(model(xtest).shape)
