import speech_recognition as sr
import librosa
import soundfile as sf
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
from librosa.core import audio
from numpy import lib
import os
import sys
sys.path.append('c:/nmb/nada/python_import/')
from volume_handling import volume_normal
import matplotlib.pyplot as plt

# origin_dir = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M_denoise\\1_file\\'
# out_dir = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M_normal\\1_file\\'
# volume_normal(origin_dir=origin_dir,out_dir=out_dir)


r = sr.Recognizer()

txt = list()
f = open('C:\\nmb\\nmb_data\\STT\\STT voice\\stt_m.txt', 'w')

path_a = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M_denoise\\1_file\\'
path_b = 'C:\\nmb\\nmb_data\\STT\\STT voice\\M_normal\\1_file\\'
path_list = [path_a, path_b]


file_list = os.listdir(path_a)
print(file_list)

for i in range(1,11):
    for j in path_list:
        infiles = librosa.util.find_files(j)
        print(infiles)
        if j == path_a:
            a = '[denoise]'
            print(a)
        elif j == path_b:
            a = '[normal]'
            print(a)
        for infile in infiles:
            _, w_id = os.path.split(infile)
            w_id = w_id[:-4]
            print(w_id)
            audio = sr.AudioFile(j + w_id + '.wav')
            with audio as audio_file:
                file = r.record(audio_file)
            stt = r.recognize_google(file, language='ko-KR')
            spelled_sent = spell_checker.check(stt)
            checked_sent = spelled_sent.checked
            f.write(a + '\n' + checked_sent + '\n')
            print(stt)
    f.write('\n')
f.close()