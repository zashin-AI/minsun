import os
import sys
sys.path.append('c:/nmb/nada/python_import/')

from feature_handling import load_data_mel, load_data_denoise_mel

# filepath = 파일 불러올 경로
# filename = 파일 확장자명 (wav, flac...)
# labels = 라벨링 (여자 0, 남자 1)

# filepath = 'c:/nmb/nmb_data/audio_data/F/'
# filename = 'wav'
# labels = 0

# data, label = load_data_mel(filepath, filename, labels)


# DENOISE 파일
filepath = 'C:\\nmb\\nmb_data\\1s2m\\pansori_m_split\\'
filename = 'wav'
labels = 1

data, label = load_data_denoise_mel(filepath, filename, labels)