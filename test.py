import torch
from torch import nn
import pandas as pd
from tools import *
from config import Params
from representation.SigRepre import MultiSignalRepresentation
from models import MER
import torch.optim as optim

args = Params(use_cuda=True, debug=True)

spliter = load_model(args.spliter)
data = pd.read_pickle(args.data)

train_index, test_index = None, None

for i, k in enumerate(spliter[args.valid]):
    args.k = i
    print("\n" + "=======" * 6 + '[Fold {}]'.format(i), "=======" * 6)
    train_index = k['train_index']
    test_index = k['test_index']
    break

dataprepare = DataPrepare(args,
                          target=args.target,
                          data=data,
                          train_index=train_index,
                          test_index=test_index,
                          device=args.device,
                          batch_size=args.batch_size)

model = MultiSignalRepresentation(seq=1536,
                                  output_size=40, pretrained=False, device=args.device)

model.output_layer = MER.MERClassifer(args, 1)
model.to(args.device)

train_dataloader, test_dataloader = dataprepare.get_data()


optimizer = optim.AdamW(model.parameters(), lr=0.001)

# train the model
for epoch in range(10):
    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = nn.BCELoss()(y_pred.squeeze(), y_batch.float().squeeze())
        loss.backward()
        for name, parms in model.named_parameters():
            print('-->name:', name,
                  '-->grad_requirs:', parms.requires_grad,
                  ' -->grad_value:', parms.grad)
            break
        optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item()}")

    # evaluate the model on test data
    test_acc = 0
    count = 0
    with torch.no_grad():
        for X_test, y_test in test_dataloader:
            y_pred = model(X_test)
            y_pred = (y_pred > 0.5).int().squeeze()
            accuracy = (y_pred == y_test.squeeze()).float().mean()
            test_acc += accuracy.item()
            count += 1

    print(f"Epoch {epoch+1},  accuracy: {test_acc/count:.4f}")
