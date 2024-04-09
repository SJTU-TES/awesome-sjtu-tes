"""
This task is running a cross validation.
We start from the two-fold validation.
"""
#%% Import necessary packages and EER function
# test the numpy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import os
import random

def eer(x_test, y_test, model):
    preds = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)


#%%
data = np.load("/home/fazhong/Github/czx/data.npy", allow_pickle=True)
labels = np.load("/home/fazhong/Github/czx/labels.npy", allow_pickle=True)


data_all = []
data = data.tolist()
#print(data[0])
labels = labels.tolist()
for i in range(len(data)):
    tmp = []
    tmp.append(np.array(data[i][0])) 
    tmp.extend([labels[i][0]])
    tmp.extend([labels[i][1]])
    tmp.extend([labels[i][2]])
    data_all.append(tmp)
random.shuffle(data_all)
data = data_all
# ?
#np.random.shuffle(data)
batch_size = 10
feature_len = 110
loss_function = binary_crossentropy
## batch
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

#%% training and save the hdf5 file
data_train = data[:int(0.5*(len(data)))]
print(len(data_train))
X1 = np.asarray([x[0] for x in data_train])
print(X1.shape)
y1 = np.asarray([x[1] for x in data_train])
print(y1.shape)
data_test = data[int(0.5*(len(data))):]
X2 = np.asarray([x[0] for x in data_test])
y2 = np.asarray([x[1] for x in data_test])
checkpointer = ModelCheckpoint(filepath="./data-task0/train1.keras",
                               verbose=verbosity, save_best_only=True)
print('-' * 30)
print('Training for whole data set')
history = model.fit(X1, y1,
                    # validation_data=(x[test], y[test]),
                    validation_split=0.1,
                    batch_size=batch_size,
                    epochs=no_epochs,
                    verbose=verbosity,
                    callbacks=[checkpointer, tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)]
                    )

## train for X2
checkpointer = ModelCheckpoint(filepath="./data-task0/train2.keras",
                               verbose=verbosity, save_best_only=True)
print('-' * 30)
print('Training for whole data set')
history = model.fit(X2, y2,
                    # validation_data=(x[test], y[test]),
                    validation_split=0.1,
                    batch_size=batch_size,
                    epochs=no_epochs,
                    verbose=verbosity,
                    callbacks=[checkpointer, tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)]
                    )


#%% calculate the final result.
#data_train = np.load("./main_task/data-task0/data1.npy", allow_pickle=True)
X1 = np.asarray([x[0] for x in data_train])
y1 = np.asarray([x[1] for x in data_train])
#data_test = np.load("./main_task/data-task0/data2.npy", allow_pickle=True)
X2 = np.asarray([x[0] for x in data_test])
y2 = np.asarray([x[1] for x in data_test])


model.load_weights("./data-task0/train1.keras")
scores = model.evaluate(X2, y2)
y_pred2 = model.predict(X2)
print(y_pred2.shape)

model.load_weights("./data-task0/train2.keras")
scores = model.evaluate(X1, y1)
y_pred1 = model.predict(X1)

y_pred = np.concatenate((y_pred1, y_pred2))
y_pred = y_pred.reshape((len(y_pred), 1))
y_label = np.concatenate((y1, y2))
y_label = y_label.reshape((len(y_label), 1))
for i in range(len(y_label)):
    if(y_pred[i]>0.5):y_pred[i]=1
    else:y_pred[i] = 0
ACCU = np.sum((y_pred == y_label)) / len(y_label)
print("ACCU is " + str(100 * ACCU))
fpr, tpr, thresholds = roc_curve(y_label, y_pred)
EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
print(EER)


# #%% calculate the final result.
# num_all = np.zeros((20, 1))
# num_success = np.zeros((20, 1))
# for user_num in range(1, 21, 1):
#     # testing the data on the train1.hdf5
#     model.load_weights("./data-task0/train1.keras")
#     print("user number is " + str(user_num))
#     X_test = np.asarray([x[0] for x in data_test if (x[5] == user_num and x[1] == 0)])
#     y_test = np.asarray([x[1] for x in data_test if (x[5] == user_num and x[1] == 0)])
#     scores = model.evaluate(X_test, y_test)
#     num_all[user_num - 1] += len(y_test)
#     num_success[user_num - 1] += np.round(len(y_test)*scores[1])
# for user_num in range(1, 21, 1):
#     # testing the data on the train2.hdf5
#     model.load_weights("./data-task0/train2.keras")
#     print("user number is " + str(user_num))
#     X_test = np.asarray([x[0] for x in data_train if (x[5] == user_num and x[1] == 0)])
#     y_test = np.asarray([x[1] for x in data_train if (x[5] == user_num and x[1] == 0)])
#     scores = model.evaluate(X_test, y_test)
#     num_all[user_num - 1] += len(y_test)
#     num_success[user_num - 1] += np.round(len(y_test)*scores[1])

# #%% show the results
# for user_num in range(1, 21, 1):
#     print("user number is " + str(user_num))
#     print("[=========]  total number is " + str(int(num_all[user_num - 1])) + ", and wrong detect " + str(int(num_all[user_num - 1] - num_success[user_num - 1]))
#           + " samples, rate is " + str(np.round(num_success[user_num - 1] / num_all[user_num - 1], 4)))
# print("total number is " + str(int(np.sum(num_all))) + ", and detect " + str(int(np.sum(num_all) - np.sum(num_success)))
#       + " samples, rate is " + str((np.sum(num_success) / np.sum(num_all))))
