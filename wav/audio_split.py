import librosa
from pydub import AudioSegment
import soundfile as sf
import os
import sys
sys.path.append('c:/nmb/nada/python_import/')

from voice_handling import voice_split, voice_split_m

# ---------------------------------------------------------------
# voice_split: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기
# def voice_split(origin_dir, threshold, out_dir):
# **** example ****
# origin_dir(하나의 wav파일이 있는 경로+파일명) = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
# threshold(몇초씩 자를지 5초는 5000) = 5000
# out_dir(5초씩 잘려진 wav 파일을 저장할 경로) = 'D:/nmb_test/test_split/'




for i in range(1, 19):
    filename = f'M ({i})_denoise'
    origin_dir = 'C:\\nmb\\nmb_data\\audio_data_denoise\\1s2m\\pansori_m_silence_denoise\\'+ filename + '.wav'
    threshold = 1000 # 몇초씩 자를 것인지 설정
    out_dir = 'C:\\nmb\\nmb_data\\1s2m\\pansori_m_split\\'
    end_threshold = 120000 # 끝나는 지점(2분)
    voice_split_m(origin_dir=origin_dir, threshold=threshold, end_threshold =end_threshold,out_dir=out_dir)


# 여러 파일 for 문 사용
# audio_dir = 'C:\\nmb\\nmb_data\\audio\\open_slr_wav\\'
# infiles = librosa.util.find_files(audio_dir)

# count = 0
# for file in infiles:
#     origin_dir = infiles[count]
#     threshold = 5000 # 몇초씩 자를 것인지 설정
#     out_dir = 'C:\\nmb\\nmb_data\\audio\\open_slr_f_2m\\'
#     end_threshold = (60*2)*1000 # 끝나는 지점(1분)
#     voice_split_1m(origin_dir=origin_dir, threshold=threshold, end_threshold =end_threshold,out_dir=out_dir)
#     count += 1