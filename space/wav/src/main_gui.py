#code
#coding=UTF-8
# ! -*- coding: utf-8 -*-
from __future__ import print_function

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

import os
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image,ImageTk
from tkinter.messagebox import *
from tkinter import scrolledtext
top1=None
top2=None
top4=None
top5=None
top6=None
top7=None
img_open=None
img=None
v1=None
v2=None
ll=0
s2=''
s1=''
top3=None
t2=None
s=''
f1="fg.txt"
f2="fg.txt"
v=None
top=None
v={}
d1={}
d2={}
message=""
ermsg=""
picn=0
arg = []
class MyThread(threading.Thread):
    def __init__(self, func, *args):#多线程启动，防止界面卡死
        super().__init__()

        self.func = func
        self.args = args

        self.setDaemon(True)
        self.start()

    def run(self):
        self.func(*self.args)

def chf(tt1):#选择音频文件
    global f1
    f1=filedialog.askopenfilename()
    showinfo("Open File", "Open a new File.")
    tt1.delete(0.0, tk.END)
    tt1.insert(0.0, f1)


def info():
    pp='语言接口安全'
    showinfo('Information',pp)

def build_model():
    # %% 定义分类器model
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
    model.load_weights("model.hdf5")
    # 从train2.hdf5导入model。
    # train2.hdf5 是从 data2.npy训练来的。
    # 这样与 data1.npy数据不会有重叠
    return model

def show_data(f1):
    file_path = f1
    print(f1)
    rate, data = read(file_path)
    plt.plot(data, label='Voice Signal')
    plt.show()


def show_feature(f1):
    file_path = f1
    file_name = file_path.split("\\")[-1]
    # define the normal or attack in variable cur_y
    if 'normal' in file_name:
        cur_y = 1  # normal case
    elif 'attack' in file_name:
        cur_y = 0
    # split the file name
    # read the data
    rate, data = read(file_path)
    X = mald_feature(rate, data)
    # 从wav 文件提取特征的函数是 generate_array_feature.py
    # X 是特征，特征的维度是110维
    y = cur_y
    # y是标签，1代表正常样本，0代表攻击样本
    if y == 1: # 正常情况
        title = 'the sample is normal'
    else:
        title = 'the sample is attack'
    plt.plot(X)
    plt.title(title)
    plt.show()


def detect(f1, model):
    file_path = f1
    file_name = file_path.split("\\")[-1]
    # define the normal or attack in variable cur_y
    if 'normal' in file_name:
        cur_y = 1  # normal case
    elif 'attack' in file_name:
        cur_y = 0
    # split the file name
    # read the data
    rate, data = read(file_path)
    X = []
    X += [list(mald_feature(rate, data))]
    X += [list(mald_feature(rate, data))]
    # 加2次，因为model需要一个二维的
    X = np.asarray(X)

    # 从wav 文件提取特征的函数是 generate_array_feature.py
    # X 是特征，特征的维度是110维
    y = cur_y
    # y是标签，1代表正常样本，0代表攻击样本
    y_pred = np.round(model.predict(X))
    # 开始预测
    y_pred = y_pred[0]

    if y == 1:  # 正常情况
        if y_pred == y:
            print("成功预测")  # 成功预测
            print("车辆运行")
            title = "指令正常，预测正确，车辆运行"
            print('--------------')
            print(title)
        else:
            print("失败预测")  # 失败预测
            print("车辆静止")
            title = "指令正常，预测失败，车辆静止"
            print('--------------')
            print(title)
    else:  # 异常情况，决策是相反的
        if y_pred == y:
            print("成功预测")  # 成功预测
            print("车辆静止")
            title = "指令异常，预测正确，车辆静止"
            print('--------------')
            print(title)
        else:
            print("失败预测")  # 失败预测
            print("车辆运行")
            title = "指令异常，预测失败，车辆运行"
            print('--------------')
            print(title)


ans=""


root=tk.Tk(className='语音接口认证系统')
#root.iconbitmap('bf.ico')
root.attributes("-alpha",0.9)
tk.Label(root,height=10,width=5).grid(row=0,column=0)
fra=tk.Frame(root,width=55,height=100)
fra.grid(row=0,column=1)
tk.Label(root,height=10,width=5).grid(row=0,column=2)
tk.Label(fra,text='',height=1,width=10).grid(row=0,column=0)

tt1=tk.Text(fra,height=2,width=30)
tt1.grid(row=1,column=0)
tk.Button(fra, text='请先选择语音数据', command=lambda: chf(tt1)).grid(row=1,column=1)
model = build_model()


train=tk.Button(fra,text='显示音频内容',font=('楷体,bold'),borderwidth=3,command=lambda :MyThread(show_data,f1))     #完成
train.grid(row=3,column=0)

train=tk.Button(fra,text='显示音频的特征',font=('楷体,bold'),borderwidth=3,command=lambda :MyThread(show_feature,f1))     #完成
train.grid(row=5,column=0)

train=tk.Button(fra,text='显示检测结果',font=('楷体,bold'),borderwidth=3,command=lambda :MyThread(detect,f1,model))     #完成
train.grid(row=7,column=0)


tk.mainloop()
