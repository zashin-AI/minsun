
# data load

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os



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
            mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
            mels = librosa.amplitude_to_db(mels, ref=np.max)
            # plt.figure(figsize=(10,4))
            # plt.title('mel spectrogram')
            # librosa.display.specshow(mels, sr=sr, x_axis='time')
            # plt.colorbar()
            # plt.show()

            dataset.append(mels)
            label.append(index)
    
    dataset = np.array(dataset)
    label = np.array(label)
    print(dataset.shape)    
    print(label.shape)      

    np.save(f'C:\\nmb\\nmb_data\\npy\\brandnew_{index}_mels.npy', arr=dataset)
    print("dataset save")
    np.save(f'C:\\nmb\\nmb_data\\npy\\brandnew_{index}_mels_label.npy', arr=label)
    print("label save")

    index += 1 


print('=====save done=====') 
# ------------------------------------------------------

# ====0=====
# (1104, 128, 862)
# (1104,)
# dataset save
# label save
# ====1=====
# (1037, 128, 862)
# (1037,)
# dataset save

# ------------------------------------------------------

F = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_0_mels.npy')
print(F.shape)  # (1104, 128, 862)
M = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_1_mels.npy')
print(M.shape)  # (1037, 128, 862)