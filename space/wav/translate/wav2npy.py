import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from sklearn.preprocessing import normalize
from generate_array_feature import mald_feature, get_filelist
import time
import os
from pydub import AudioSegment
import whisper
folder_path = '/home/fazhong/Github/czx2/example/data'  
names = ['feng','jc','meng','zhan']
types = ['01','02','03','04','05','06','07','08','09','09','10','11','12','13','14','15','16','17','18','19','20']
voice = []

def convert_6ch_wav_to_stereo(input_file_path, output_file_path):
    sound = AudioSegment.from_file(input_file_path, format="wav")
    if sound.channels != 6:
        raise ValueError("The input file does not have 6 channels.")
    front_left = sound.split_to_mono()[0]
    front_right = sound.split_to_mono()[1]
    center = sound.split_to_mono()[2]
    back_left = sound.split_to_mono()[4]
    back_right = sound.split_to_mono()[5]
    center = center - 6  
    back_left = back_left - 6  
    back_right = back_right - 6  
    stereo_left = front_left.overlay(center).overlay(back_left)
    stereo_right = front_right.overlay(center).overlay(back_right)
    stereo_sound = AudioSegment.from_mono_audiosegments(stereo_left, stereo_right)
    stereo_sound.export(output_file_path, format="wav")

def read_all_files(directory):
    data = []
    labels = []
    texts = []
    whisper_model = whisper.load_model("large")
    out_path='/home/fazhong/Github/czx/temp/temp.wav'
    i=0
    for root, dirs, files in os.walk(directory):
        
        for file in files:
            #if i > 10:return data,labels,texts
            content = []
            content_label = []
            file_path = os.path.join(root, file)
            convert_6ch_wav_to_stereo(file_path,out_path)
            result = whisper_model.transcribe(out_path,language="en")
            text_result = result['text']
            texts.append(text_result)
            print(file)
            if 'normal' in file:
                label = 1  # normal case
            elif 'attack' in file:
                label = 0
            for name in names:
                if name in file:
                    name_index = names.index(name)
            if label == 0:
                category_number = int(file.split('_')[4])
            elif label == 1:
                category_number = int(file.split('_')[3])

            rate, wavdata = read(file_path)
            content.append(list(mald_feature(rate, wavdata)))
            content_label.append(label)
            content_label.append(name_index)
            content_label.append(category_number)
            data.append(content)
            labels.append(content_label)
            i+=1
    return data,labels,texts

# 调用函数
data,labels,texts = read_all_files(folder_path)
data_array = np.array(data)
labels_array = np.array(labels)
texts_array = np.array(texts)
filename = 'data.npy'
filename2 = 'labels.npy'
filename3 = 'texts.npy'
np.save(filename, data_array)
np.save(filename2, labels_array)
np.save(filename3, texts_array)
print('fin')
# #%% 导入音频
# path_wave = r"/home/fazhong/Github/czx/voice"
# print("Loading data ...")
# name_all = get_filelist(path_wave)
# voice = []
# # voice 是从 一堆 wav 音频文件中提取的波形
# X = []  # X is the feature ~ data[0]
# y = []  # y is the normal (1) or attack (0) ~ data[1]

# for file_path in name_all:
#     file_name = file_path.split("\\")[-1]
#     # define the normal or attack in variable cur_y
#     if 'normal' in file_name:
#         cur_y = 1  # normal case
#     elif 'attack' in file_name:
#         cur_y = 0
#     # split the file name
#     # read the data
#     rate, data = read(file_path)
#     voice += [list(data)]

#     X += [list(mald_feature(rate, data))]
#     y += [cur_y]

# norm_X = normalize(X, axis=0, norm='max')
# X = np.asarray(norm_X)
# y = np.asarray(y)