# data load(현민)

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os

# 정규화 (MinMaxScaler)
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


dataset = []
label = []
pathAudio_F = 'C:\\nmb\\nmb_data\\brandnew_data\\F\\'
pathAudio_M = 'C:\\nmb\\nmb_data\\brandnew_data\\M\\'

files_F = librosa.util.find_files(pathAudio_F, ext=['flac','wav'])
files_M = librosa.util.find_files(pathAudio_M, ext=['flac','wav'])

files_F = np.array(files_F)
files_M = np.asarray(files_M)

print(files_F.shape)    # (1200,)
print(files_M.shape)    # (1200,)

total = [files_F, files_M]
index = 0               # index 0 : 여성, 1 : 남성

for folder in total : 
    print(f"===={index}=====")
    dataset = []
    label = []
    for file in folder:
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        length = (len(y) / sr)
        if length < 5.0 : pass
        else:
            mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20, n_fft=512, hop_length=128)
            mfccs = normalize(mfccs, axis=1)

            # plt.figure(figsize=(10,4))
            # plt.title('MFCCs')
            # librosa.display.specshow(mfccs, sr=sr, x_axis='time')
            # plt.colorbar()
            # plt.show()

            dataset.append(mfccs)
            label.append(index)
    
    dataset = np.array(dataset)
    label = np.array(label)
    print(dataset.shape)    
    print(label.shape)      

    np.save(f'C:\\nmb\\nmb_data\\npy\\brandnew_{index}_mfccs.npy', arr=dataset)
    print("dataset save")
    np.save(f'C:\\nmb\\nmb_data\\npy\\brandnew_{index}_mfccs_label.npy', arr=label)
    print("label save")

    index += 1 


print('=====save done=====') 
# ------------------------------------------------------

# ====0=====
# (1104, 20, 862)
# (1104,)
# dataset save
# label save
# ====1=====
# (1037, 20, 862)
# (1037,)
# dataset save

# ------------------------------------------------------

F = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_0_mfccs.npy')
print(F.shape)  # (1104, 20, 862)
M = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_1_mfccs.npy')
print(M.shape)  # (1037, 20, 862)