# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2023-01-16 15:30:02

# from CONSTANT import *

from param import Params
from tools import *

from SigRepre import MultiSignalRepresentation

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import math
import datetime
import os
import sys
import time
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

torch.manual_seed(3407)


def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_normal_(m.weight)


def run(args,
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_data,
        val_data=None,
        ckpt_path='checkpoint.pt',
        patience=5,
        monitor="val_loss",
        mode="min"):
    history = {}
    lrs = []

    best_result = None

    if args.init:
        model.apply(init_xavier)

    for epoch in range(1, args.epochs + 1):
        printlog("Epoch {0} / {1}".format(epoch,
                                          args.epochs))
        # training -------------------------------------------------
        train_step_runner = StepRunner(model=model,
                                       loss_fn=loss_fn,
                                       metrics=args.metrics,
                                       stage='train',
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(steprunner=train_step_runner,
                                         metrics=args.metrics)

        train_metrics = train_epoch_runner(train_data)
        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(model=model,
                                         loss_fn=loss_fn,
                                         metrics=args.metrics_val,
                                         stage='val')
            val_epoch_runner = EpochRunner(val_step_runner, args.metrics)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for metric_name, metric in val_metrics.items():
                history[metric_name] = history.get(metric_name, []) + [metric]

            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step(val_metrics['val_loss'])

        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(
            arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor, arr_scores[best_score_idx]))
            best_result = arr_scores[best_score_idx]
            if not args.debug:
                torch.save(model.state_dict(), ckpt_path)
        if len(arr_scores) - best_score_idx > patience:
            print(
                "<<<<<< {} without improvement in {} epoch, early stopping >>>>>>"
                .format(monitor, patience))
            break
        if not args.debug:
            model.load_state_dict(torch.load(ckpt_path))

    history = pd.DataFrame(history)
    history['lr'] = lrs
    print()
    print(history)
    return history, {monitor: best_result}


def main():
    st = time.time()
    args = Params()
    dataprepare = DataPrepare(
        args, datapath=r'../processed_signal/all_384_12s_step_2s.pkl')
    train_dataloader, test_dataloader = dataprepare.get_data()

    model = MultiSignalRepresentation(
        output_size=40, device=args.device, seq=384)
    model = model.to(args.device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=5,
                                  verbose=True,
                                  threshold_mode='rel',
                                  cooldown=0,
                                  min_lr=0,
                                  eps=1e-08)

    history_df, best_result = run(args,
                                  model,
                                  optimizer,
                                  scheduler,
                                  loss_fn,
                                  patience=24,
                                  train_data=train_dataloader,
                                  val_data=test_dataloader, ckpt_path=args.checkpoint)
    time_used = time.time() - st
    print()
    print(best_result)
    print('[Used time: {}s]'.format(round(time_used), 4))
    if not args.debug:
        history_df.to_csv(
            os.path.join(args.save_path, 'history_df.csv'), index=False)

    print(args.save_path)


if __name__ == '__main__':
    main()
