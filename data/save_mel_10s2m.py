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

# female
# pathAudio_F = 'C:\\nmb\\nmb_data\\1s2m\\open_slr_f_split\\'
# load_data_mel(pathAudio_F, 'wav', 0)

# male
# pathAudio_M = 'C:\\nmb\\nmb_data\\1s2m\\corpus_m_split\\'
# load_data_mel(pathAudio_M, 'wav', 1)

################################ pansori ##################################
# 여자 39명 (화자 수 맞추기 위해 44명에서 5명 뺌)
x1 = np.load('C:\\nmb\\nmb_data\\npy\\10s2m\\open_slr_f_data.npy')
y1 = np.load('C:\\nmb\\nmb_data\\npy\\10s2m\\open_slr_f_label.npy')
print(x1.shape) 
print(y1.shape)
print(y1[:10]) 
# (468, 128, 1723)
# (468,)
# [0 0 0 0 0 0 0 0 0 0]

# 남자 40명
x2 = np.load('C:\\nmb\\nmb_data\\npy\\10s2m\\open_slr_m_data.npy')
y2 = np.load('C:\\nmb\\nmb_data\\npy\\10s2m\\open_slr_m_label.npy')
print(x2.shape) 
print(y2.shape)
print(y2[:10]) 
# (480, 128, 1723)
# (480,)
# [1 1 1 1 1 1 1 1 1 1]

