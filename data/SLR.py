
import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os



dataset = []
label = []
pathAudio_F = 'C:\\nmb\\nmb_data\\SLR\\F\\'
pathAudio_M = 'C:\\nmb\\nmb_data\\SLR\\M\\'

files_F = librosa.util.find_files(pathAudio_F, ext=['flac','wav'])
files_M = librosa.util.find_files(pathAudio_M, ext=['flac','wav'])
print(files_F)
print(files_M)
files_F = np.array(files_F)
files_M = np.asarray(files_M)


# y, sr = librosa.load(audio_path)
# total = [files_F, files_M]
# index = 0               # index 0 : 여성, 1 : 남성


# for file in folder:
#     y, sr = librosa.load(file)
#     length = (len(y) / sr)
#     print('길이 : ', length)
 
    

# print(files_F.shape)    # (1200,)
# print(files_M.shape)    # (1200,)
# print('오디오 길이(초) : %.2f' % (len(y) / sr))