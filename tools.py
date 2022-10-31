from scipy import special
from scipy.signal import freqs
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle


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
