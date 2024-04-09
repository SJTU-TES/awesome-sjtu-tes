"""
This python script is used for direction detection.
We design the direction detection for 3 wav files types which has
4, 6 and 8 channels.
"""
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, freqz
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

offsetVector = []

"""
The funtion butter_highpass, butter_highpass_filter and calculateResidues are shared
Function offset is used for getAngle_for_eight
"""
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def calculateResidues(Chan1, Chan2, fs):
    S1 = butter_highpass_filter(Chan1, 100, fs, 7)
    S2 = butter_highpass_filter(Chan2, 100, fs, 7)

    index1 = -1
    index2 = -1
    index = -1

    for i in range(len(S1)):
        if S1[i] > 0.03:
            index1 = i
            break

    for i in range(len(S2)):
        if S2[i] > 0.03:
            index2 = i
            break

    if (index1 < index2):
        index = index1
    else:
        index = index2

    residues = np.mean(np.square(S1[index:index + 401] - S2[index:index + 401]))
    # offsetVector.append( index1 )

    return residues


def do_iac(signal, pairs, fs):
    # signal = data / 32767
    residuesVector = []

    for offset in [5, -5]:

        # Computer overall cancellation error for this angle
        iterator = 0
        residues = 0
        for mic1, mic2 in pairs:

            Chan1 = signal[:, mic1]
            Chan2 = signal[:, mic2]

            S1 = Chan1  # butter_highpass_filter(Chan1 , 100 , fs , 7)
            S2 = Chan2  # butter_highpass_filter(Chan2 , 100 , fs , 7)

            index = -1
            for i in range(len(S1)):
                if (S1[i] > 0.003 and i > 40):
                    index = i
                    break

            if (iterator == 0 or iterator == 4):
                a = S1[index - 15:index + 15]
                b = S2[index - 15:index + 15]
                residues += np.square(np.subtract(a, b))
            elif (iterator == 1 or iterator == 3):
                a = S1[index - 15 + offset // 2:index + 15 + offset // 2]
                b = S2[index - 15:index + 15]
                residues += np.square(np.subtract(a, b))
            elif (iterator == 2):
                a = S1[index - 15 + offset:index + 15 + offset]
                b = S2[index - 15:index + 15]
                residues += np.square(np.subtract(a, b))
            elif (iterator == 5 or iterator == 7):
                a = S1[index - 15 - offset // 2:index + 15 - offset // 2]
                b = S2[index - 15:index + 15]
                residues += np.square(np.subtract(a, b))
            elif (iterator == 6):
                a = S1[index - 15 - offset:index + 15 - offset]
                b = S2[index - 15:index + 15]
                residues += np.square(np.subtract(a, b))

            iterator += 1

        residuesVector.append(np.mean(residues))

    return residuesVector[0] < residuesVector[1]


def calculateResidues_eight(Chan1, Chan2, fs):
    S1 = Chan1  # butter_highpass_filter(Chan1 , 100 , fs , 7 )
    S2 = Chan2  # butter_highpass_filter(Chan2 , 100 , fs , 7 )

    index1 = -1
    index2 = -1
    index = -1

    for i in range(len(S1)):
        if S1[i] > 0.01:
            index1 = i
            break

    for i in range(len(S2)):
        if S2[i] > 0.01:
            index2 = i
            break

    if (index1 < index2):
        index = index1
    else:
        index = index2

    residues = np.mean(np.square(S1[index:index + 401] - S2[index:index + 401]))

    return residues


def getAngle_for_eight(data, fs):
    signal = data / 32767
    for i in range(8):
        column = butter_highpass_filter(signal[:, i], 100, fs, 7)
        signal[:, i] = column

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]
    smallestResidues = 100
    closestPair = (0, 0)
    offsetIndex = -1

    for iter in range(8):

        chan1 = signal[:, pairs[iter][0]]
        chan2 = signal[:, pairs[iter][1]]

        residues = calculateResidues_eight(chan1, chan2, fs)

        if (residues < smallestResidues):
            smallestResidues = residues
            closestPair = (pairs[iter])
            offsetIndex = iter

    if do_iac(signal, pairs, fs) == True:
        d1 = abs(offsetIndex - 4)
        d2 = abs((offsetIndex + 4) % 8 - 4)
        if (d1 < d2):
            pass
        else:
            closestPair = pairs[(offsetIndex + 4) % 8]

    mics = (closestPair[0] + 1, closestPair[1] + 1)

    return mics


def getAngle_for_six(data, fs):
    signal = data / 32767
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    smallestResidues = 100
    closestPair = (0, 0)
    offsetIndex = -1

    for iter in range(6):

        chan1 = signal[:, pairs[iter][0]]
        chan2 = signal[:, pairs[iter][1]]

        residues = calculateResidues(chan1, chan2, fs)

        if (residues < smallestResidues):
            smallestResidues = residues
            closestPair = (pairs[iter])
            offsetIndex = iter

    """ if (offsetVector[offsetIndex] > offsetVector[(offsetIndex+3)%6]   ):
        closestPair = pairs[(offsetIndex+3)%6] """

    mics = (closestPair[0] + 1, closestPair[1] + 1)
    # print(offsetVector)

    return mics


def getAngle_for_four(data, fs):
    signal = data / 32767
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    smallestResidues = 100
    closestPair = (0, 0)
    offsetIndex = -1

    for iter in range(4):

        chan1 = signal[:, pairs[iter][0]]
        chan2 = signal[:, pairs[iter][1]]

        residues = calculateResidues(chan1, chan2, fs)

        if (residues < smallestResidues):
            smallestResidues = residues
            closestPair = (pairs[iter])
            offsetIndex = iter

    """ if (offsetVector[offsetIndex] > offsetVector[(offsetIndex+3)%6]   ):
        closestPair = pairs[(offsetIndex+3)%6] """

    mics = (closestPair[0] + 1, closestPair[1] + 1)
    # print(offsetVector)

    return mics


def getDirection_Pair(closestPair, num_chan):
    """
    :param closestPair: two closet pair, such as (0,1)
    :param num_chan: channel numbers, such as 8
    :return: in above parameters, should be [7 0 1 2]
    """
    pairs = [0, 0, 0, 0]
    pairs[1] = closestPair[0] - 1
    pairs[2] = closestPair[1] - 1
    pairs[0] = (pairs[1] - int(num_chan/2)) % num_chan
    pairs[3] = (pairs[2] + int(num_chan/2)) % num_chan

    return pairs



