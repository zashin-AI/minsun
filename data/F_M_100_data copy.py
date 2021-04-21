
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
pathAudio_F = 'C:\\nmb\\nmb_data\\predict\\gan'


files_F = librosa.util.find_files(pathAudio_F, ext=['flac','wav'])

for file in files_F:
    y, sr = librosa.load(file, sr=22050, duration=5.0)
    mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
    mels = librosa.amplitude_to_db(mels, ref=np.max)
    model.predict

files_F = np.array(files_F)

print(files_F.shape)    # (100,)

total = [files_F]
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
    
    dataset = np.array(dataset[:100])
    label = np.array(label[:100])
    print(dataset.shape)    
    print(label.shape)      

    np.save(f'C:\\nmb\\nmb_data\\npy\\M100_{index}_mels.npy', arr=dataset)
    print("dataset save")
    np.save(f'C:\\nmb\\nmb_data\\npy\\M100_{index}_mels_label.npy', arr=label)
    print("label save")

    index += 1 


print('=====save done=====') 
# ------------------------------------------------------
# F100
# ====0=====
# (89, 128, 862)
# (89,)
# dataset save
# label save  
# ====1=====  
# (1037, 128, 862)
# (1037,)
# dataset save
# label save
# 1,126

# M100
# ====0=====
# (1104, 128, 862)
# (1104,)
# dataset save
# label save
# ====1=====
# (69, 128, 862)
# (69,)
# dataset save
# label save
# 1,173
# ------------------------------------------------------

F = np.load('C:\\nmb\\nmb_data\\npy\\M100_0_mels.npy')
print(F.shape)  # (1104, 128, 862)
M = np.load('C:\\nmb\\nmb_data\\npy\\M100_1_mels.npy')
print(M.shape)  # (1037, 128, 862)