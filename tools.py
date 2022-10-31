from scipy import special
from scipy.signal import freqs
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


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
