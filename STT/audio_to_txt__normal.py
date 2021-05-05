from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence      # 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import speech_recognition as sr #pip install SpeechRecognition
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
import librosa.display
import librosa


r = sr.Recognizer()

# volume_file = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\corpus\\normal\\', ext=['wav'])
volume_file = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\STT_M_pred\\M_normal\\1_file\\', ext=['wav'])

path_list = []
volume = []
for j, path in enumerate(volume_file) : 
    sound_file = AudioSegment.from_wav(path)
    dbfs = sound_file.dBFS
    thresh = int(dbfs)
    if dbfs < thresh :
        thresh = thresh - 1
    audio_chunks = split_on_silence(sound_file,  
        min_silence_len= 500,
        silence_thresh= dbfs - 16,
        keep_silence= 300
    )
    path_list.append(path)

    full_txt = []
    for i, chunk in enumerate(audio_chunks):    
        out_file = "C:\\nmb\\nmb_data\\chunk\\test\\"+ str(j) + f"chunk{i}.wav"
        chunk.export(out_file, format="wav")
        aaa = sr.AudioFile(out_file)
        with aaa as source :
            audio = r.record(aaa)
        try:
            txt = r.recognize_google(audio, language="ko-KR")
            spelled_sent = spell_checker.check(txt)
            checked_sent = spelled_sent.checked
            full_txt.append(str(checked_sent)) # 하나로 합칠 경우 사용
        except : # 너무 짧은 음성은 pass 됨 
            pass   
        print(checked_sent)
        volume.append(checked_sent)  
    
print(path_list)
print('볼륨 : ', volume)


# #----------------------------------------------------------------------

# pairs = dict(zip(origin, speed))        # key=value 형태로 dict 를 생성
# lines = map(lambda item:'원본 :{}\n 볼륨 :{}\n'.format(item[0], item[1]), pairs.items())       
# # 코드:{};비용:{}; 형태의 템플릿으로 저장할 문자열 리스트를 생성
# with open('C:\\nmb\\nmb_data\\STT\\test.txt', 'wt') as f: f.writelines(lines) 


for k in range(len(path_list)) :
    path_list[k]

new = ''
for i in range(len(volume)):
    new += volume[i] + '\n'

new_sum = path_list[k] + checked_sent

with open('C:\\nmb\\nmb_data\\STT\\text_test.txt', 'wt') as f:
     f.writelines(new_sum)        



