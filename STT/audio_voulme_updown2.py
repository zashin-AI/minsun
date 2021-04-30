from pydub import AudioSegment, effects
import os,librosa, librosa.display
import sys
sys.path.append('c:/nmb/nada/python_import/')
from volume_handling import volume_updown
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence      # 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import speech_recognition as sr #pip install SpeechRecognition
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
import librosa.display
import librosa

# path = "C:\\nmb\\nmb_data\\STT\\STT voice\\M_denoise\\112_003_0107_denoise.wav"
# outpath = "C:\\nmb\\nmb_data\\volume\\updown\\M_pred\\"

# list = [10,15,20,25,30]
# for i in list:
#     volume = int(i)
#     volume_updown(path=path, volume=volume, outpath=outpath)

#=================================================================================
r = sr.Recognizer()

file_list = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\STT voice\\M_denoise\\1_file\\', ext=['wav'])
print(file_list)

for j, path in enumerate(file_list) : 

    # 오디오 불러오기
    sound_file = AudioSegment.from_wav(path)
    # 가장 최소의 dbfs가 무엇인지
    # dbfs : 아날로그 db과는 다른 디지털에서의 db 단위, 0일 때가 최고 높은 레벨
    dbfs = sound_file.dBFS
    # print(sound_file.dBFS)
    thresh = int(dbfs)
    # print(int(sound_file.dBFS))

    # 최소의 dbfs를 threshold에 넣는다.
    if dbfs < thresh :
        thresh = thresh - 1
        # print(thresh)

    # silence 부분 마다 자른다. 
    audio_chunks = split_on_silence(sound_file,  
        # # split on silences longer than 1000ms (1 sec)
        # min_silence_len=500,
        # # anything under -16 dBFS is considered silence
        # silence_thresh= thresh , 
        # # keep 200 ms of leading/trailing silence (음성의 앞, 뒤 갑자기 뚝! 끊기는 걸 방지하기 위한 기능인 것 같음)
        # keep_silence=200

        min_silence_len= 100,
        silence_thresh= dbfs - 100,
        keep_silence= 300
    )

    full_txt = []
    # 말 자른 거 저장 & STT 
    for i, chunk in enumerate(audio_chunks):    
        out_file = "C:\\nmb\\nmb_data\\chunk\\test\\"+ str(j) + f"chunk{i}.wav"
        # print ("exporting", out_file)
        chunk.export(out_file, format="wav")
        aaa = sr.AudioFile(out_file)
        with aaa as source :
            audio = r.record(aaa)
        # print(type(audio))
        try:
            txt = r.recognize_google(audio, language="ko-KR")
            # print(txt)

            # 한국어 맞춤법 체크
            spelled_sent = spell_checker.check(txt)
            checked_sent = spelled_sent.checked
            # print(checked_sent)

            # full_txt.append(str(txt))
            full_txt.append(str(checked_sent))
            # print(txt)
            # print(full_txt)
        except : # 너무 짧은 음성은 pass 됨 
            pass
    print("파일 이름 : ",path[path.rfind('\\') + 1:])
    # rfind를 사용하여 오른쪽부터 '\\'의 인덱스를 찾고, +1부터 문자열의 마지막까지 가져온다.
    for checked_sent in full_txt:
        print(checked_sent)
