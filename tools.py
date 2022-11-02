from scipy import special
from scipy.signal import freqs
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import torch
from torch.utils.data import (
    TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler)


def join_signals(df, target='valence'):
    bvp_cols = [fea for fea in df.columns.values if fea.split('_')[0] in [
        'BVP']]
    eda_cols = [fea for fea in df.columns.values if fea.split('_')[0] in [
        'EDA']]
    temp_cols = [fea for fea in df.columns.values if fea.split('_')[0] in [
        'TEMP']]
    hr_cols = [fea for fea in df.columns.values if fea.split('_')[0] in ['HR']]

    target_cols = ['valence', 'arousal', 'arousal_rating', 'valence_rating']
    group_cols = ['participant_id', 'song_id']

    signal_concats = []
    for bvp, eda, temp, hr in zip(df[bvp_cols].values, df[eda_cols].values, df[temp_cols].values, df[hr_cols].values):
        signal_concats.append([bvp, eda, temp, hr])

    return np.array(signal_concats), df[target].values


def resample_by_poly(signal, input_fs, output_fs):
    return signal.resample_poly(signal, input_fs, output_fs)


def resample_by_interpolation(signal, input_fs, output_fs):
    scale = output_fs / input_fs
    n = round(len(signal) * scale)
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, len(signal), endpoint=False),
        signal,
    )
    return resampled_signal


def segment_generator(signal, win_size, step=1):
    return [signal[i:i+win_size] for i in range(0, len(signal)-win_size+1, step)]


def plot_sig(sig):
    fig = plt.figure(figsize=(25, 5))
    plt.plot(sig)
    plt.show()


def chauvenet_filter(signal):
    mean, std, N = signal.mean(), signal.std(), len(signal)
    criterion = 1.0 / (2 * N)
    d = abs(signal - mean) / std
    prob = special.erfc(d)

    mask = prob < criterion

    signal = pd.Series(np.ma.masked_array(data=signal, mask=mask,
                                          fill_value=np.nan).filled())

    return signal.interpolate()


def iqr_filter(signal):
    q75, q25 = np.percentile(signal, [75, 25])
    intr_qr = q75 - q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    signal[signal < min] = np.nan
    signal[signal > max] = np.nan

    return signal.interpolate()


def butter_bandpass(lowcut, highcut, fs, order=4):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=True)
    return b, a


def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def minmax_scale(signal):
    return (signal - signal.min()) / (signal.max() - signal.min())


def get_info(filename):
    filename, extension = os.path.split(filename)
    return filename


def get_folder_files(folderpath):
    folderfiles = []
    for root, dirs, files in os.walk(folderpath):
        for file in files:
            folderfiles.append(os.path.join(root, file))
    return folderfiles


def reduce_mem_usage(df, un_process, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    orig_cols = len(df.columns.values)
    for col in df.columns:
        if col in un_process:
            continue
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    print('Final', df.shape)
    return df


def writetxt(filaname, line):
    with open(filaname, 'a') as f:
        f.write(line)


def save_model(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(filename, 'saved done!')


def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


class DataPrepare(object):
    def __init__(self, target, data, train_index, test_index, device, batch_size=64):

        X, y = join_signals(data, target=target)
        xtrain, ytrain, xtest, ytest = X[train_index], y[train_index], X[test_index], y[test_index]
        print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

        self.xtrain = torch.from_numpy(xtrain).to(device).to (torch.float32)
        self.xtest = torch.from_numpy(xtest).to(device).to (torch.float32)

        self.ytrain = torch.from_numpy(ytrain).to(device).to (torch.float32)
        self.ytest = torch.from_numpy(ytest).to(device).to (torch.float32)

        print(self.xtrain.isnan().any(), self.xtest.isnan().any(),
              self.ytrain.isnan().any(), self.ytest.isnan().any(),)

        self.batch_size = batch_size

    def get_data(self):
        train_data = TensorDataset(self.xtrain, self.ytrain)
        test_data = TensorDataset(self.xtest, self.ytest)

        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_size, drop_last=True)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=self.batch_size, drop_last=True)

        return train_dataloader, test_dataloader
