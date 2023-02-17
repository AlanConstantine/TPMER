import torch
from torch import nn
import torch.nn.functional as F
from representation.SigRepre import MultiSignalRepresentation


class MERClassifer(nn.Module):
    def __init__(self, args, n_class, output_size=160) -> None:
        super().__init__()
        # self.pretrain_model = MultiSignalRepresentation(
        #     output_size=40, device=args.device)
        # self.pretrain_model.load_state_dict(
        #     torch.load(r'./representation/mask08ep13.pt'))
        # self.encoder = self.pretrain_mode.encoder
        self.fcn = nn.Sequential(
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_class)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.fcn(x)


class MERRegressor(nn.Module):
    def __init__(self, output_size=160) -> None:
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.fcn(x)
