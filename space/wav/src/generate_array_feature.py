'''
This is the main ArrayID feature building script

revised: April 04, 2021

'''

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft, fftfreq
from scipy import signal
import random
from librosa.core import lpc
import librosa.feature
import csv
from sklearn.preprocessing import normalize
from direction_detection import *


##############################################
# HELPER FUNCTIONS

# converts hz to indices -> allows splicing of freq data
def hz_to_indices(freqs, lowcut, highcut):
    i = 0
    while freqs[i] < lowcut:
        i += 1
    low = i
    while freqs[i] < highcut:
        i += 1
    return low, i


# compresses our feature vectors
# After extracting our features, they could be different lengths depending on
# the input signal, so we normalize each feature vector to be the same no matter
# the speaker
def get_row_compressor(old_dimension, new_dimension):
    dim_compressor = np.zeros((new_dimension, old_dimension))
    bin_size = float(old_dimension) / new_dimension
    next_bin_break = bin_size
    which_row = 0
    which_column = 0
    while which_row < dim_compressor.shape[0] and which_column < dim_compressor.shape[1]:
        if round(next_bin_break - which_column, 10) >= 1:
            dim_compressor[which_row, which_column] = 1
            which_column += 1
        elif next_bin_break == which_column:

            which_row += 1
            next_bin_break += bin_size
        else:
            partial_credit = next_bin_break - which_column
            dim_compressor[which_row, which_column] = partial_credit
            which_row += 1
            dim_compressor[which_row, which_column] = 1 - partial_credit
            which_column += 1
            next_bin_break += bin_size
    dim_compressor /= bin_size
    return dim_compressor

# helper functions for above function
def get_column_compressor(old_dimension, new_dimension):
    return get_row_compressor(old_dimension, new_dimension).transpose()

def compress_and_average(array, new_shape):
    return np.mat(get_row_compressor(array.shape[0], new_shape[0])) * \
           np.mat(array) * \
           np.mat(get_column_compressor(array.shape[1], new_shape[1]))
##############################################


##############################################
# MAIN FEATURE EXTRACTION FUNCTIONS


def get_filelist(dir):
    Filelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


def lpcc(data, n=15):
    """
    f_LPC = lpcc(data, n): get the LPCC from the voice data
    The order n is 15
    """
    size_lpc = n  # define the order of LPCC
    a = lpc(data, order = size_lpc)  # use the built-in function
    a = -a
    f_LPC = np.zeros(len(a))
    f_LPC[0] = np.log(size_lpc)
    for i in range(1, len(a)):
        k = np.arange(1, i)  # k from 1 to i-1
        f_LPC[i] = a[i] + np.sum((1 - k/i) * a[k] * f_LPC[i - k])
    return f_LPC[1:]


# returns long term fft
def get_ltfd(spec, m=20, start_index=1, end_index=86):
    # only get the useful part
    spec = spec[:, start_index: end_index, :(spec.shape[2] - spec.shape[2] % m)]

    # merge the spec in the time line
    channels = np.sum(spec, axis=2)

    all_ffts = np.sum(channels, axis=0)
    all_ffts /= np.max(all_ffts)

    channels_ffts = np.asarray([channels[i, :] / np.max(channels[i, :]) for i in range(channels.shape[0])])

    return all_ffts, channels_ffts


# returns long term fft
def get_ltfp(spec, m=20, start_index_fp=1, end_index_fp=86):
    # only get the useful part
    spec = spec[:, start_index_fp:end_index_fp, :(spec.shape[2] - spec.shape[2] % m)]

    # split the data
    splices = np.asarray(np.split(spec, m, axis=2))

    # merge the data (wang ge hua)
    mesh = np.zeros((splices.shape[0], splices.shape[1], splices.shape[2]))
    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            for k in range(mesh.shape[2]):
                mesh[i, j, k] = np.sum(splices[i, j, k, :])

    # calculate the standard deviation
    std_feature = np.zeros((mesh.shape[0], mesh.shape[2]))
    for i in range(std_feature.shape[0]):
        for j in range(std_feature.shape[1]):
            std_feature[i, j] = np.std(mesh[i, :, j]) / np.mean(mesh[i, :, j])

    # define the ltfp
    LTFP = np.mean(std_feature, axis=0)
    LTFP = LTFP / np.max(LTFP)
    return LTFP


def feature_distribution(channel_fft):
    num_feature = 5
    f_dis = np.zeros(2 * num_feature)
    co = np.zeros((num_feature, len(channel_fft)))
    for num in range(len(channel_fft)):
        a = channel_fft[num]
        for i in range(1, len(a)):
            a[i] = a[i-1] + a[i]
        a = a / np.max(a)
        dis_index = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i in range(len(dis_index)):
            co[i, num] = find_value(a, dis_index[i])
        co[:, num] /= len(a)
    for i in range(num_feature):
        f_dis[i] = np.mean(co[i, :])
        f_dis[i + num_feature] = np.std(co[i, :])
    return co, f_dis


def find_value(a, dis_index):
    c = 0
    for i in range(len(a) - 1):
        if a[i] <= dis_index <= a[i + 1]:
            c = i
            break
    return c


def mald_feature(rate, data):
    n_fft = 4096
    # detect the direction
    if data.shape[1] == 4:
        closestPair = getAngle_for_four(data, fs=rate)
    elif data.shape[1] == 6:
        closestPair = getAngle_for_six(data, fs=rate)
    elif data.shape[1] == 8:
        closestPair = getAngle_for_eight(data, fs=rate)
    pairs = getDirection_Pair(closestPair, data.shape[1])

    # low and high thresholds for field print features -> we want 1 - 10kHz range
    lowcut_fp = 1
    highcut_fp = 5000
    if highcut_fp > rate / 2:  # in case the sampling rate is very small
        highcut_fp = rate / 2 - 100
    highcut_fd = 1000

    # input rate -> make sure to change this based on device.
    # All of the devices are 44100 except for the AMLOGIC, which is 16kHz.
    # If this rate is not changed acccordingly, the _ltfp and _ltfft features
    # will be off

    # just some helper splicing globals
    freq = fftfreq(n_fft, 1. / rate)  # data = logmmse(data, rate)
    start_index, end_index = hz_to_indices(freq, lowcut_fp, highcut_fd)
    start_index_fp, end_index_fp = hz_to_indices(freq, lowcut_fp, highcut_fp)


    # empty feature vectors
    _lpcc = []
    # extract lfp and lpcc from each channel independently, then sum
    for i in pairs:
        a = np.asfortranarray(data[:, i]).astype(dtype=float)
        _lpcc += list(lpcc(a))

    # calculate the spectrogram
    spec = [signal.stft(data[:, i], fs=rate, window='hann', nperseg=1024, noverlap=768, nfft=n_fft)[2] for i in range(data.shape[1])]
    spec = np.asarray(spec)  # convert list to numpy
    # obtain the absolute value
    spec = np.abs(spec)

    # get the ltfp feature


    # get ltfp features and compress to a 50 feature vectoc
    _ltfd, channel_fft = get_ltfd(spec=spec, start_index=start_index, end_index=end_index)

    _ltfd = list(compress_and_average(_ltfd.reshape(len(_ltfd), 1), (20, 1)).flat)

    co, _fdis = feature_distribution(channel_fft)

    # get ltfp features and compress to a 50 feature vector
    _ltfp = get_ltfp(spec=spec, start_index_fp=start_index_fp, end_index_fp=end_index_fp)
    _ltfp = list(compress_and_average(_ltfp.reshape(len(_ltfp), 1), (20, 1)).flat)

    # out is final feature vector, each data point formed as a tuple : (X, y), where X is the feature vector and y is the label
    # X_y is just compiled l ist of all the tuples
    feature = np.concatenate((_lpcc, _ltfd, _fdis, _ltfp))
    return feature

