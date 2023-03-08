# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2022-11-16 00:35:04

from argparse import ArgumentParser
# from torchmetrics.functional import auc, mean_squared_error
# from torchmetrics import F1Score
from tools import *
from CONSTANT import *
# from models import CNNBiLSTM, CNNTransformer, SigRep
from models import MER
from config import Params
from torch.utils.data import (TensorDataset, DataLoader, SequentialSampler,
                              WeightedRandomSampler)
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time
import datetime
import os
import sys
import time
import warnings
from copy import deepcopy
from representation.SigRepre import MultiSignalRepresentation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import gc

# warnings.filterwarnings('ignore')

torch.cuda.empty_cache()
torch.manual_seed(3407)
# torch.cuda.set_device(1)


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "======" * 6 + "[%s]" % nowtime + "======" * 6)
    print(str(info) + "\n")


def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_normal_(m.weight)


class StepRunner:

    def __init__(self,
                 net,
                 loss_fn,
                 args,
                 stage="train",
                 metrics_dict=None,
                 optimizer=None):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer = optimizer
        self.args = args
        self.results = None
        self.sig = nn.Sigmoid()

    def step(self, features, labels):
        preds = self.net(features)
        # print(preds)

        if self.optimizer is not None and self.stage == "train":
            self.optimizer.zero_grad()

        if self.args.target in ['valence_rating', 'arousal_rating']:
            loss = self.loss_fn(preds, labels.float())
        else:
            print(preds.shape, labels.long().shape)
            loss = self.loss_fn(preds, labels.long().reshape(-1, ))

        if self.stage == "train":
            loss.backward()
            self.optimizer.step()

        # metrics
        step_metrics = {}
        clf_reports = []
        confu_mat = None
        for name, metric_fn in self.metrics_dict.items():

            if self.args.target in ['valence_rating', 'arousal_rating']:
                step_metrics[self.stage + "_" + name] = metric_fn(
                    preds, labels).item()
            else:
                _, predicted = torch.max(preds.data, 1)
                # print(predicted.shape)
                if name == 'f1':
                    step_metrics[self.stage + "_" + name] = metric_fn(
                        predicted, labels.long().reshape(-1, )).item()
                # elif name == 'auc':
                #     step_metrics[self.stage + "_" + name] = metric_fn(
                #         torch.round(self.sig(preds)).long(), labels).item()
                elif name == 'acc':
                    step_metrics[self.stage + "_" + name] = metric_fn(
                        predicted, labels.long().reshape(-1, )).item()
                    # step_metrics[self.stage + "_" +
                    #              name] = (predicted == labels).sum().item() / preds.shape[0]
                else:
                    pass
                if self.stage != 'train':
                    clf_reports.append(classification_report(labels.cpu().detach().numpy(), predicted.cpu().detach().numpy(
                    ), output_dict=True, zero_division=1))
                    confu_mat = confusion_matrix(labels.cpu().detach().numpy(), predicted.cpu().detach().numpy(
                    ), )
        self.results = step_metrics
        if self.stage != 'train':
            clf_reports[-1] = confu_mat
        return loss.item(), step_metrics, clf_reports

    def train_step(self, features, labels):
        self.net.train()
        return self.step(features, labels)

    @ torch.no_grad()
    def eval_step(self, features, labels):
        self.net.eval()
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:

    def __init__(self, steprunner, args):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.args = args

    def __call__(self, dataloader):
        total_loss, step = 0, 0

        epoch_metrics = {
            self.stage + '_' + name: 0
            for name in list(self.args.metrics_dict)
        }

        epoch_clf_reports = []

        loop = tqdm(enumerate(dataloader),
                    total=len(dataloader),
                    file=sys.stdout)
        for i, batch in loop:
            loss, step_metrics, clf_reports = self.steprunner(*batch)

            if len(clf_reports) != 0:
                epoch_clf_reports.append(clf_reports)

            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            for name, scores in step_metrics.items():
                epoch_metrics[name] += scores

            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step

                for name, scores in step_metrics.items():
                    epoch_metrics[name] = round(epoch_metrics[name] / step, 4)

                epoch_log = dict({self.stage + "_loss": epoch_loss},
                                 **epoch_metrics)
                loop.set_postfix(**epoch_log)
            gc.collect()

        epoch_log.update(epoch_metrics)

        return epoch_log, epoch_clf_reports


def train_model(args,
                net,
                optimizer,
                scheduler,
                loss_fn,
                metrics_dict,
                train_data,
                val_data=None,
                epochs=10,
                ckpt_path='checkpoint.pt',
                patience=5,
                monitor="val_loss",
                mode="min"):

    history = {}
    lrs = []

    best_result = None
    best_score_idx = 0
    final_rep = []

    if args.init:
        net.apply(init_xavier)

    for epoch in range(1, epochs + 1):
        printlog("[Fold {0}] Epoch {1} / {2}".format(args.k, epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(net=net,
                                       stage="train",
                                       loss_fn=loss_fn,
                                       args=args,
                                       metrics_dict=deepcopy(metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(
            train_step_runner, args)
        train_metrics, train_epoch_clf_reports = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(args=args,
                                         net=net,
                                         stage="val",
                                         loss_fn=loss_fn,
                                         metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(
                val_step_runner, args)
            with torch.no_grad():
                val_metrics, test_epoch_clf_reports = val_epoch_runner(
                    val_data)

                final_rep.append(test_epoch_clf_reports)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step(val_metrics['val_loss'])

        if args.show_wei:
            for name, parms in net.named_parameters():
                if name in ['cnns.0.weight', 'lstm2.bias_hh_l1']:
                    print('\t', name, torch.mean(parms.data),
                          parms.requires_grad, torch.mean(parms.grad))

            # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(
            arr_scores)

        if best_score_idx == len(arr_scores) - 1:
            print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor, arr_scores[best_score_idx]))
            best_result = arr_scores[best_score_idx]
            if not args.debug:
                torch.save(net.state_dict(), ckpt_path)
        if len(arr_scores) - best_score_idx > patience:
            print(
                "<<<<<< {} without improvement in {} epoch, early stopping >>>>>>"
                .format(monitor, patience))
            print('Current best results', best_result)
            break
        if not args.debug:
            # print('loading best checkpoint:', ckpt_path)
            net.load_state_dict(torch.load(ckpt_path))

    final_rep = final_rep[best_score_idx]
    history = pd.DataFrame(history)
    history['lr'] = lrs
    return history, {monitor: best_result}, final_rep


def run(train_dataloader, test_dataloader, args):
    model = None
    if args.pretrain:
        model = MER.SignalSample(input_size=args.input_size)
        # model = MultiSignalRepresentation(
        #     output_size=40, device=args.device, pretrain=True)
        # model.load_state_dict(torch.load(args.pretrain))

        # loss_fn = nn.BCEWithLogitsLoss()

        # if args.target in ['valence_rating', 'arousal_rating']:
        #     loss_fn = nn.MSELoss()
        #     mode = "min"
        #     model.fcn = MER.MERRegressor()
        # else:
        #     model.fcn = MER.MERClassifer(args, 2)
        rep = MultiSignalRepresentation(seq=1536,
                                        output_size=40, device=args.device, pretrained=True)
        rep.load_state_dict(torch.load(args.pretrain))
        rep.output_layer = MER.MERClassifer(args, 2)
        model.output_layer = rep
    else:
        print('Learning from Scratch......')
        model = MultiSignalRepresentation(seq=1536,
                                          output_size=40, device=args.device, pretrained=True)
        model.output_layer = MER.MERClassifer(args, 2)

    loss_fn = nn.CrossEntropyLoss()
    mode = 'max'
    # device_ids = [0, 1]
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    model = model.to(args.device)
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
    metrics_dict = args.metrics_dict

    history_df, best_result, final_rep = train_model(
        args,
        model,
        optimizer,
        scheduler,
        loss_fn,
        metrics_dict,
        train_data=train_dataloader,
        val_data=test_dataloader,
        epochs=args.epochs,
        patience=24,
        monitor="val_{}".format(list(metrics_dict.keys())[0]),
        mode=mode,
        ckpt_path=os.path.join(
            args.save_path, 'fold{}_{}'.format(str(args.k), 'checkpoint.pt')))
    return history_df, best_result, final_rep


def main():

    args = Params()
    print('\n'.join("%s: %s" % item for item in vars(args).items()))

    spliter = load_model(args.spliter)
    data = pd.read_pickle(args.data)

    for i, k in enumerate(spliter[args.valid]):
        st = time.time()
        args.k = i
        print("\n" + "=======" * 6 + '[Fold {}]'.format(i), "=======" * 6)
        train_index = k['train_index']
        test_index = k['test_index']
        dataprepare = DataPrepare(args,
                                  target=args.target,
                                  data=data,
                                  train_index=train_index,
                                  test_index=test_index,
                                  device=args.device,
                                  batch_size=args.batch_size)
        train_dataloader, test_dataloader = dataprepare.get_data()
        history_df, best_result, final_rep = run(
            train_dataloader, test_dataloader, args)

        time_used = time.time() - st
        args.results[args.k] = {
            'history': history_df,
            'best_result': best_result,
            'time_used': time_used,
            'clf_rep': final_rep
        }
        print()
        print('[Used time: {}s]'.format(round(time_used), 4))

        if args.debug:
            break
    if not args.debug:
        args.save_results(results=args.results)

    print(args.save_path)
    avg_res = []
    for fold in args.results.keys():
        # if fold == 'valid_clf_report':
        #     continue
        print('Fold', fold, 'best result:', args.results[fold]['best_result'],
              'Time used:', args.results[fold]['time_used'])
        avg_res.append(args.results[fold]['best_result'][list(
            args.results[fold]['best_result'].keys())[0]])
    print('Avg. result: ', np.mean(avg_res))


def api(args):
    print('\n'.join("%s: %s" % item for item in vars(args).items()))

    spliter = load_model(args.spliter)
    data = pd.read_pickle(args.data)

    for i, k in enumerate(spliter[args.valid]):
        st = time.time()
        args.k = i
        print("\n" + "=======" * 6 + '[Fold {}]'.format(i), "=======" * 6)
        train_index = k['train_index']
        test_index = k['test_index']
        dataprepare = DataPrepare(args,
                                  target=args.target,
                                  data=data,
                                  train_index=train_index,
                                  test_index=test_index,
                                  device=args.device,
                                  batch_size=args.batch_size)
        train_dataloader, test_dataloader = dataprepare.get_data()
        history_df, best_result, final_rep = run(
            train_dataloader, test_dataloader, args)

        time_used = time.time() - st
        args.results[args.k] = {
            'history': history_df,
            'best_result': best_result,
            'time_used': time_used,
            'clf_rep': final_rep
        }
        print()
        print('[Used time: {}s]'.format(round(time_used), 4))

        if args.debug:
            break
    if not args.debug:
        args.save_results(results=args.results)

    print(args.save_path)
    avg_res = []
    for fold in args.results.keys():
        # if fold == 'valid_clf_report':
        #     continue
        print('Fold', fold, 'best result:', args.results[fold]['best_result'],
              'Time used:', args.results[fold]['time_used'])
        avg_res.append(args.results[fold]['best_result'][list(
            args.results[fold]['best_result'].keys())[0]])
    print('Avg. result: ', np.mean(avg_res))
    return np.mean(avg_res)


if __name__ == '__main__':
    main()
