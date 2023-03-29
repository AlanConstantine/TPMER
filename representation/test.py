import torch
from torch import nn
import pandas as pd
from tools import *
from param import Params
# from SigRepre import MultiSignalRepresentation

from PhySiRES import MultiSignalRepresentation

args = Params(use_cuda=True, debug=True)
# dataprepare = DataPrepare(args)

model = MultiSignalRepresentation(seq=768,
                                  output_size=40, pretrained=False, device=args.device)
model.to(args.device)
# xtest = dataprepare.X_test[:2]
# del dataprepare
xtest = torch.rand((2, 4, 768)).to(args.device)
# print(xtest.shape)
print(model(xtest).shape)

# 384-8
#
