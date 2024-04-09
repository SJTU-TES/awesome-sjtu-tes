import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import os
import random
import spacy
import matplotlib.pyplot as plt
# == Part 1 - Read data ==

data = np.load("/home/fazhong/Github/czx/data.npy", allow_pickle=True)
labels = np.load("/home/fazhong/Github/czx/labels.npy", allow_pickle=True)
texts = np.load("/home/fazhong/Github/czx/texts.npy", allow_pickle=True)
commands = [
    "OK Google.",
    "Turn on Bluetooth.",
    "Record a video.",
    "Take a photo.",
    "Open music player.",
    "Set an alarm for 6:30 am.",
    "Remind me to buy coffee at 7 am.",
    "What is my schedule for tomorrow?",
    "Square root of 2105?",
    "Open browser.",
    "Decrease volume.",
    "Turn on flashlight.",
    "Set the volume to full.",
    "Mute the volume.",
    "What's the definition of transmit?",
    "Call Pizza Hut.",
    "Call the nearest computer shop.",
    "Show me my messages.",
    "Translate please give me directions to Chinese.",
    "How do you say good night in Japanese?"
]
commands_basic = [
    0,# "OK Google.",
    1,#"Turn on Bluetooth.",
    5,#"Set an alarm for 6:30 am.",
    10,#"Decrease volume.",
    11,#"Turn on flashlight.",
    12,#"Set the volume to full.",
    13,#"Mute the volume.",
]
commands_daily = [
    2,#"Record a video.",
    3,#"Take a photo.",
    4,#"Open music player.",
    6,#"Remind me to buy coffee at 7 am.",
    15,#"Call Pizza Hut.",

]
commands_work = [
    7,#"What is my schedule for tomorrow?",
    8,#"Square root of 2105?",
    9,#"Open browser.",
    14,#"What's the definition of transmit?",
    16,#"Call the nearest computer shop.",
    17,#"Show me my messages.",
    18,#"Translate please give me directions to Chinese.",
    19,#"How do you say good night in Japanese?"
]

def rule_judge(type,time,location):
    if type in commands_basic:
        if time == 0:
            return False
        else:
            return True
    elif type in commands_daily:
        if time == 2:
            return True
        else:
            return False
    elif type in commands_work:
        if time == 1 and location ==1:
            return True
        else:
            return False

# 0 - sleep time / 1 - work time / 2 - daily time
times_label = [0,1,2]
# 0 - home / 1 - factory
location_label = [0,1]

data_all = []
data = data.tolist()
labels = labels.tolist()
texts = texts.tolist()

acc_num = 0
all_num = len(data)
atk_list = []
atk_err = []
name_err = []
type_err = []

gt_label = []
pre_label = []

name_err_num = [0,0,0,0]
name_acc_num = [0,0,0,0]
command_err_num = []
command_acc_num = []
for i in range(20):
    command_err_num.append(0)
    command_acc_num.append(0)

for i in range(len(data)):
    tmp = []
    tmp.append(np.array(data[i][0])) 
    tmp.extend([labels[i][0]])
    tmp.extend([labels[i][1]])
    tmp.extend([labels[i][2]])
    data_all.append(tmp)
data = data_all

time_labels = []
location_labels = []
for i in range(len(data)):
    time_labels.append(random.randint(0,2))
    location_labels.append(random.randint(0,1))

rule_err = []

for i in range(len(data)):
    if not rule_judge(data[i][2],time_labels[i],location_labels[i]):
        rule_err.append(i)

#  == Part 2 - Judge of Human == 
model = load_model('/home/fazhong/Github/czx/data-task0_1/train1.keras')
X = np.asarray([x[0] for x in data])
y = np.asarray([x[1] for x in data])
type = np.asarray([x[3] for x in data])

y_pred = model.predict(X)
y_pred = y_pred.reshape((len(y_pred), 1))
y = y.reshape((len(y), 1))
for i in range(len(y)):
    if(y_pred[i]>0.5):y_pred[i]=1
    else:
        y_pred[i] = 0
        atk_list.append(i)
    if(y_pred[i]!=y[i]):
        atk_err.append(i)
ACCU = np.sum((y_pred == y)) / len(y)
print(len(y))
print("ACCU is " + str(100 * ACCU))

#  == Part 3 - Judge of Name ==

model = load_model('/home/fazhong/Github/czx/data-task0/train1.keras')
y_name = np.asarray([x[2] for x in data])
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred,axis=1)
ACCU = np.sum((y_pred_classes == y_name)) / len(y_name)
for i in range(len(y_name)):
    if(y_pred_classes[i]!=y_name[i]):
        name_err.append(i)
print("ACCU is " + str(100 * ACCU))


# Part 4 - Transcribe and Judge of Reason

# PS! Attack的文本不需要跑分类
nlp = spacy.load('en_core_web_md')


def classify_key(command):
    if 'ok google' in command:
            return 1
    elif 'okay' in command:
            return 1
    elif 'bluetooth' in command:
        return 2
    elif 'record' in command and 'video' in command:
        return 3
    elif 'take' in command and 'photo' in command:
        return 4
    elif 'music' in command:
        return 5
    elif 'alarm' in command:
        return 6
    elif 'remind' in command and 'coffee' in command:
        return 7
    elif 'am' in command :
        return 7
    elif 'schedule' in command or 'tomorrow' in command:
        return 8
    elif 'square root' in command:
        return 9
    elif 'open browser' in command:
        return 10
    elif 'decrease volume' in command:
        return 11
    elif 'flashlight' in command and 'on' in command:
        return 12
    elif 'hello freshlight' in command.lower():
        return 12
    elif 'turn on' in command:
        return 12
    elif 'volume' in command and 'full' in command:
        return 13
    elif 'mute' in command :
        return 14
    elif 'move' in command :
        return 14
    elif 'more' in command :
        return 14
    elif 'motor' in command :
        return 14
    elif 'mood' in command :
        return 14
    elif 'most' in command :
        return 14
    elif 'what' in command :
        return 14
    elif 'with' in command :
        return 14
    elif 'milk' in command :
        return 14
    elif 'use' in command :
        return 14
    elif 'definition of' in command:
        return 15
    elif 'call' in command and 'pizza hut' in command.lower():
        return 16
    elif 'copies are' in command.lower() or 'call a piece of heart' in command.lower() or 'copies of' in command.lower():
        return 16
    elif 'peace' in command.lower():
        return 16
    elif 'heart' in command.lower():
        return 16
    elif 'pisa' in command.lower():
        return 16
    elif 'piece' in command.lower():
        return 16
    elif 'hard' in command.lower():
        return 16
    elif 'call' in command and 'computer shop' in command.lower():
        return 17
    elif 'message' in command :
        return 18
    elif 'translate' in command:
        return 19
    elif 'good night' in command and 'in japanese' in command:
        return 20
    else:
        return None  # or some default value if command is not recognized

correct_count = 0
total_count = 0
category_number = 0
total_normal = 0

normal_texts = []
normal_labels = []

All_Normal_names = []

# Test of rule module
test_flag = True
atk_org_list = []
for i in range(len(texts)):
    if test_flag:
        normal_texts.append(texts[i])
        All_Normal_names.append(y_name[i])
        normal_labels.append(type[i])
        if y[i] == 0:
            atk_org_list.append(i)
    else:
        if y[i] == 1:
            normal_texts.append(texts[i])
            All_Normal_names.append(y_name[i])
            normal_labels.append(type[i])

print(len(atk_org_list))
# for text in texts:
#     if texts.index(text) in atk_list:
#         print(texts.index(text))
#         continue
#     else:
#         normal_texts.append(text)

weird_name = []
weird_command = []


# for i in range(len(data)):
#     if not rule_judge(data[i][2],time_labels[i],location_labels[i]):
#         rule_err.append(i)

for i in range(len(normal_texts)):
    text = normal_texts[i]
    category_number = normal_labels[i]
    # print(text)
    # print(category_number)

    result_pre = classify_key(text.replace('.', '').replace(',', '').lower().strip())

    # IF rule - judge

    # if not rule_judge(category_number-1,time_labels[i],location_labels[i]):
    #     command_err_num[category_number-1]+=1
    #     name_err_num[All_Normal_names[i]]+=1
    #     continue
    if i in atk_org_list:
        command_err_num[category_number-1]+=1
        name_err_num[All_Normal_names[i]]+=1
        continue
    if result_pre is not None:
        if result_pre  == category_number:
            correct_count += 1
            command_acc_num[category_number-1]+=1
            name_acc_num[All_Normal_names[i]]+=1
            continue
    input_doc = nlp(text.replace('.', '').replace(',', '').lower().strip())
    similarities = [(command, input_doc.similarity(nlp(command))) for command in commands]
    best_match = max(similarities, key=lambda item: item[1])
    best_match_index = commands.index(best_match[0]) + 1
    if best_match_index == category_number:
        correct_count += 1
        command_acc_num[category_number-1]+=1
        name_acc_num[All_Normal_names[i]]+=1
    else:
        # print(text.replace('.', '').replace(',', '').lower().strip())
        # if category_number==16:
        #     print(input_doc,commands[category_number-1],commands[best_match_index-1])
        command_err_num[category_number-1]+=1
        name_err_num[All_Normal_names[i]]+=1
        
        
        # if 'thank' in str(input_doc):
        #     pass
        #     # print('?')
        #     # print(texts.index(text))
        #     # print(data[texts.index(text)])
        # weird_name.append(y_name[texts.index(text)])
        # weird_command.append(type[texts.index(text)])
        type_err.append(texts.index(text))

# 计算正确率
accuracy = correct_count / len(normal_texts)
print(f"Accuracy: {accuracy:.2f}")


# Part 5 - Results
atk_set = set(atk_err)
name_set = set(name_err)
type_set = set(type_err)
#rule_set = set(rule_err)
err_list = list(atk_set | name_set | type_set)  


print(len(err_list))
# print(weird_name)

print(name_err_num)
print(name_acc_num)
print(command_err_num)
print(command_acc_num)

# print(weird_command)
#print(atk_list)
# print(len(atk_list))
# print(all_num)
# print(atk_err)
# print(name_err)
# print(type_err)
# print(type_set)
# print(err_list)

# # 设置柱状图的位置编号
# x = np.arange(len(name_err_num))

# # 画柱状图
# plt.bar(x - 0.2, name_acc_num, width=0.4, label='Correct', color='green')
# plt.bar(x + 0.2, name_err_num, width=0.4, label='Error', color='red')

# # 添加标题和标签
# plt.xlabel('Names')
# plt.ylabel('Counts')
# plt.title('Accuracy and Errors by Name')
# plt.xticks(x, ['User1', 'User2', 'User3', 'User4']) # 假设有四个名字
# plt.legend()
# #plt.savefig('/home/fazhong/Github/czx/user.png')
# # 显示图形
# plt.close()


# # 设置柱状图的位置编号
# x = np.arange(len(command_err_num))

# # 画柱状图
# plt.bar(x - 0.2, command_acc_num, width=0.4, label='Correct', color='blue')
# plt.bar(x + 0.2, command_err_num, width=0.4, label='Error', color='orange')

# # 添加标题和标签
# plt.xlabel('Commands')
# plt.ylabel('Counts')
# plt.title('Accuracy and Errors by Command')
# plt.xticks(x, [i for i in range(20)]) # 假设有六个命令
# plt.legend()

# # 显示图形
# #plt.savefig('/home/fazhong/Github/czx/com.png')