"""
这是带注释的，我用中文写了
"""
#%% 导入必要的包
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


#%% 定义分类器model
# 这一个代码块是用来定义model的。
# 定义model的batch_size, feature长度之类的
batch_size = 10
feature_len = 110
loss_function = binary_crossentropy
no_epochs = 150
optimizer = Adam()
verbosity = 1
model = Sequential()
model.add(Dense(64, input_dim=feature_len, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
# 至此，分类器模型的基本参数已经设置完毕，接下来可以从hdf5文件中导入预先训练好的model
model.load_weights(r"/home/fazhong/Github/czx/model.hdf5")
# 从train2.hdf5导入model。
# train2.hdf5 是从 data2.npy训练来的。
# 这样与 data1.npy数据不会有重叠


#%% 导入音频


data_npy = np.load('./data.npy',allow_pickle=True)
labels_npy = np.load('./labels.npy',allow_pickle=True)

data = data_npy.tolist()
labels_org = labels_npy.tolist()
labels = []
for x in labels_org:
    labels.append(x[0])


voice = []
# voice 是从 一堆 wav 音频文件中提取的波形
X = []  # X is the feature ~ data[0]
y = []  # y is the normal (1) or attack (0) ~ data[1]

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
#     print(list(mald_feature(rate, data)))
#     # 从wav 文件提取特征的函数是 generate_array_feature.py
#     # X 是特征，特征的维度是110维
#     y += [cur_y]
#     # y是标签，1代表正常样本，0代表攻击样本


X = data
Y = labels
# normalization
norm_X = normalize(X, axis=0, norm='max')

X = np.asarray(norm_X)
y = np.asarray(y)

#%% 开始预测
scores = model.evaluate(X, y)  # 这是一个总体的预测
y_pred = np.round(model.predict(X))  # 这里会给出一个预测的结论
print(y_pred)
acc = 0
for i in range(len(y)):
    if y_pred[i] == y:  acc+=1
print(acc/len(y))