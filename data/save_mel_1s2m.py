# data load

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os
import sys
import numpy as np
sys.path.append('C:/nmb/nada/python_import/')
from feature_handling import load_data_mel

# 정규화 (MinMaxScaler)

# female
# pathAudio_F = 'C:\\nmb\\nmb_data\\1s2m\\open_slr_f_split\\'
# load_data_mel(pathAudio_F, 'wav', 0)

# male
# pathAudio_M = 'C:\\nmb\\nmb_data\\1s2m\\corpus_m_split\\'
# load_data_mel(pathAudio_M, 'wav', 1)

################################ pansori ##################################
# 남자 18명 
x1 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\pansori_m_data.npy')
y1 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\pansori_m_label.npy')
print(x1.shape) 
print(y1.shape)
print(y1[:10]) 
# (2144, 128, 173) 원래 2160개인데 2분미만 파일 2개 포함해서 줄어듬
# (2144,)
# [1 1 1 1 1 1 1 1 1 1]

# 여자 10명
x2 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\pansori_f_data.npy')
y2 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\pansori_f_label.npy')
print(x2.shape) 
print(y2.shape)
print(y2[:10]) 
# (1200, 128, 173)
# (1200,)
# [0 0 0 0 0 0 0 0 0 0]

################################ slr ##################################
# 남자 40명
x3 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\slr_m_data.npy')
y3 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\slr_m_label.npy')
print(x3.shape) 
print(y3.shape)
print(y3[:10]) 
# (4800, 128, 173)
# (4800,)
# [1 1 1 1 1 1 1 1 1 1]

# 여자 39명
x4 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\slr_f_data.npy')
y4 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\slr_f_label.npy')
print(x4.shape) 
print(y4.shape)
print(y4[:10]) 
# (4680, 128, 173)
# (4680,)
# [0 0 0 0 0 0 0 0 0 0]

################################ corpus ##################################
# 남자 20명
x5 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\corpus_m_data.npy')
y5 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\corpus_m_label.npy')
print(x5.shape) 
print(y5.shape)
print(y5[:10]) 
# (2400, 128, 173)
# (2400,)
# [1 1 1 1 1 1 1 1 1 1]

# 여자 19명
x6 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\corpus_f_data.npy')
y6 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\corpus_f_label.npy')
print(x6.shape) 
print(y6.shape)
print(y6[:10]) 
# (2280, 128, 173)
# (2280,)
# [0 0 0 0 0 0 0 0 0 0]

################################ mindslab ##################################
# 남자 2명
x7 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\mindslab_m_data.npy')
y7 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\mindslab_m_label.npy')
print(x7.shape) 
print(y7.shape)
print(y7[:10]) 
# (240, 128, 173)
# (240,)
# [1 1 1 1 1 1 1 1 1 1]

# 여자 12명
x8 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\mindslab_f_data.npy')
y8 = np.load('C:\\nmb\\nmb_data\\npy\\1s2m\\mindslab_f_label.npy')
print(x8.shape) 
print(y8.shape)
print(y8[:10]) 
# (1440, 128, 173)
# (1440,)
# [0 0 0 0 0 0 0 0 0 0]

