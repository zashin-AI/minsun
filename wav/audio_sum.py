import librosa
from pydub import AudioSegment
import soundfile as sf
import os
import sys
sys.path.append('c:/nmb/nada/python_import/')
from voice_handling import voice_sum

# ---------------------------------------------------------------
# voice_sum: 오디오 한 wav 파일로 합쳐서 저장하기
# def voice_sum(form, pathaudio, save_dir, out_dir):
# **** example ****
# form(파일 형식): 'wav' or 'flac'
# audio_dir(여러 오디오가 있는 파일경로) = 'C:/nmb/nmb_data/F1F2F3/F3/'
# save_dir(flac일 경우 wav파일로 저장할 경로) = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'
# out_dir(wav파일을 합쳐서 저장할 경로+파일명까지) = "C:/nmb/nmb_data/combine_test/F3_sum.wav"

# # 1) wav일 때
a = ['6rmA7IKw7ZWY','6rmA7JiB7ZWY', '6rmA7KCV7ZiE','107','112','114','117','118','120','125']
for i in a:
    filename = f'{i}'
    path_wav = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M_wav\\'+ filename + '\\'
    path_out = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M_sum\\' + filename + '.wav'
    voice_sum(form='wav', audio_dir=path_wav, save_dir=None, out_dir=path_out)


# 2) flac일 때

# for i in range(1,11):
#     # filename = f'F ({i})'
#     filename = '6rmA7IKw7ZWY'

#     path_flac = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M\\'+ filename + '\\'
#     path_save = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M\\M_wav\\'+ filename + '\\'
#     path_out = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M\\M_sum\\' + filename + '.wav'
#     voice_sum(form='flac', audio_dir=path_flac, save_dir=path_save, out_dir=path_out)


# for i in range(1, 33):
#     form = 'flac'   
#     audio_dir = 'c:/nmb/nmb_data/SLR/F/F' + str(i) + '/'
#     save_dir = 'c:/nmb/nmb_data/SLR/save_wav/F' + str(i) + '/'
#     out_dir = 'c:/nmb/nmb_data/SLR/open_slr_f_1m/F' + str(i) + '.wav'

# voice_sum(
#         form = form,
#         audio_dir = audio_dir,
#         save_dir = save_dir,
#         out_dir = out_dir
#     )
