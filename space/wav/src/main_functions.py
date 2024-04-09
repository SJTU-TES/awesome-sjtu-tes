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
path_wave = r"/home/fazhong/Github/czx/voice"
print("Loading data ...")
name_all = get_filelist(path_wave)
voice = []
# voice 是从 一堆 wav 音频文件中提取的波形
X = []  # X is the feature ~ data[0]
y = []  # y is the normal (1) or attack (0) ~ data[1]

for file_path in name_all:
    file_name = file_path.split("\\")[-1]
    # define the normal or attack in variable cur_y
    if 'normal' in file_name:
        cur_y = 1  # normal case
    elif 'attack' in file_name:
        cur_y = 0
    # split the file name
    # read the data
    rate, data = read(file_path)
    voice += [list(data)]

    X += [list(mald_feature(rate, data))]
    print(list(mald_feature(rate, data)))
    # 从wav 文件提取特征的函数是 generate_array_feature.py
    # X 是特征，特征的维度是110维
    y += [cur_y]
    # y是标签，1代表正常样本，0代表攻击样本

# normalization
norm_X = normalize(X, axis=0, norm='max')
# X_y = [(norm_X[i], y[i]) for i in range(len(norm_X))]
# # print(len(X_y))
# # for i in X_y: print(i[1])
# X_y = np.asarray(X_y)

X = np.asarray(norm_X)
y = np.asarray(y)

# X = np.asarray([x[0] for x in X_y])
# y = np.asarray([x[1] for x in X_y])

#%% 画出特征来
index1 = [5]  # 选第2121个元素
x1 = X[index1]
y1 = y[index1]  # 1，代表normal
plt.plot(x1.T, label='normal')
index2 = [1]  # 选择第10个元素
x2 = X[index2]
y2 = y[index2]  # 0, 代表attack
plt.plot(x2.T, label='attack')
plt.legend()
plt.show()
# 可以明显看出 normal 与 attack 的区别，这也是我们分类的基础

#%% 开始预测
scores = model.evaluate(X, y)  # 这是一个总体的预测
y_pred = np.round(model.predict(X))  # 这里会给出一个预测的结论
index1 = 8  # 8 是一个正常样本
index3 = [1, 3, 5, 7, 9]  # 选一些样本，等wav 文件到了，输入就直接是wav
for i in index3:
    print('Starting detection:')
    plt.plot(voice[i], label='Voice Signal')
    plt.show()
    time.sleep(2)
    if y[i] == 1:  # 正常情况
        print('the ' + str(i) + ' sample is normal')
        title = 'the ' + str(i) + ' sample is normal'
        plt.subplot(1, 2, 1)
        plt.plot(X[index1])
        plt.subplot(1, 2, 2)
        plt.plot(X[i], label='New')
        plt.title(title)
        plt.show()
        time.sleep(1)
        if y_pred[i] == y[i]:
            print("Successfully Detect")  # 成功预测
            print("Run the car")
            title = "Successfully Detect, " + "Run the car"
            plt.title(title)
            plt.show()
        else:
            print("Detection is false.")  # 失败预测
            print("Don't run the car")
            title = "Detection is false, " + "Don't run the car"
            plt.title(title)
            plt.show()
    else:  # 异常情况，决策是相反的
        print('the ' + str(i) + ' sample is attack')
        title = 'the ' + str(i) + ' sample is attack'
        plt.subplot(1, 2, 1)
        plt.plot(X[index1], label='Normal')
        plt.subplot(1, 2, 2)
        plt.plot(X[i], label='New')
        plt.title(title)
        plt.show()
        time.sleep(1)
        if y_pred[i] == y[i]:
            print("Successfully Detect")  # 成功预测
            print("Don't run the car")
            title = "Successfully Detect, " + "Don't run the car"
            plt.title(title)
            plt.show()
        else:
            print("Detection is false.")  # 失败预测
            print("Run the car")
            title = "Detection is false, " + "Run the car"
            plt.title(title)
            plt.show()

    print("-------------------------")
