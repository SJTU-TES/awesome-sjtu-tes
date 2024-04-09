import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from sklearn.preprocessing import normalize
from generate_array_feature import mald_feature, get_filelist
import time
from pydub import AudioSegment
import whisper
import os
import spacy

# To deal with one wav file.

def is_command_reasonable(command, time, location):

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


    # Time : Work-0 / Rest-1 / Sleep-2
    # Location : Work-0 / Home-1

    commands_daily  = [
        "Call Pizza Hut.",
        "Remind me to buy coffee at 7 am.",
        "Open music player.",
        "Record a video.",
        "Take a photo.",
    ]
    commands_work = [
        "Open browser.",
        "What is my schedule for tomorrow?",
        "Square root of 2105?",
        "Call the nearest computer shop.",
        "Show me my messages.",
        "Translate please give me directions to Chinese.",
        "How do you say good night in Japanese?",
        "What's the definition of transmit?",
    ]
    commands_basic = [
        "OK Google.",
        "Decrease volume.",
        "Turn on Bluetooth.",
        "Turn on flashlight.",
        "Set the volume to full.",
        "Mute the volume.",
        "Set an alarm for 6:30 am."]


    if time == 0 and location == 0:
        if command in commands_daily:
            return False
        else:
            return True
    elif time ==2:
        if command in commands_basic:
            return True
        else:
            return False
    else:
        if command in commands_work:
            return False
        else:
            return True

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

def judge_human(rate,data):
    model = load_model('/home/fazhong/Github/czx/data-task0_1/train1.keras')
    feature =list(mald_feature(rate, data))
    features=np.array([feature])
    y_pred = model.predict(features)
    return y_pred[0]

def judge_name(rate,data):
    model = load_model('/home/fazhong/Github/czx/data-task0/train1.keras')
    feature =list(mald_feature(rate, data))
    features=np.array([feature])
    y_pred = model.predict(features)
    y_pred_classes = np.argmax(y_pred,axis=1)
    return y_pred_classes[0]

def judge_command(file_path):
    whisper_model = whisper.load_model("large")
    out_path='/home/fazhong/Github/czx/temp/temp.wav'
    convert_6ch_wav_to_stereo(file_path,out_path)
    # print(out_path)
    result = whisper_model.transcribe(out_path,language="en")
    text_result = result['text']
    print(text_result)
    return text_result

def judge_classifier(command):
    nlp = spacy.load('en_core_web_md')
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
        "What’s the definition of transmit?",
        "Call Pizza Hut.",
        "Call the nearest computer shop.",
        "Show me my messages.",
        "Translate please give me directions to Chinese.",
        "How do you say good night in Japanese?"
    ]
    def classify_key(command):
        if 'ok google' in command:
                return 1
        elif 'bluetooth' in command and 'on' in command:
            return 2
        elif 'record' in command and 'video' in command:
            return 3
        elif 'take' in command and 'photo' in command:
            return 4
        elif 'music player' in command and 'open' in command:
            return 5
        elif 'set' in command and 'alarm' in command:
            return 6
        elif 'remind' in command and 'coffee' in command:
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
        elif 'volume' in command and 'full' in command:
            return 13
        elif 'mute' in command and 'volume' in command:
            return 14
        elif 'definition of' in command:
            return 15
        elif 'call' in command and 'pizza hut' in command.lower():
            return 16
        elif 'call' in command and 'computer shop' in command.lower():
            return 17
        elif 'messages' in command and 'show' in command:
            return 18
        elif 'translate' in command:
            return 19
        elif 'good night' in command and 'in japanese' in command:
            return 20
        else:
            return None  # or some default value if command is not recognized

    file_content = command
    result_pre = classify_key(file_content.replace('.', '').replace(',', '').lower().strip())
    if result_pre is not None:
        return result_pre
    input_doc = nlp(file_content.replace('.', '').replace(',', '').lower().strip())
    similarities = [(command, input_doc.similarity(nlp(command))) for command in commands]
    best_match = max(similarities, key=lambda item: item[1])
    return best_match[0]

def judge(file_path,time,location):

    rate, data = read(file_path)
    # Maybe change to paths?
    temp = judge_human(rate,data)
    temp2 = judge_name(rate,data)
    command = judge_command(file_path)
    text = judge_classifier(command)
    if is_command_reasonable(text, time, location):
        return True
    else:
        return False

if __name__ == "__main__":
    judge('/home/fazhong/Github/czx2/example/data/fengattack60/feng_attack_echo_60_01_3.150-4.000.wav',0,0)