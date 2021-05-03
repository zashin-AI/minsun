from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence      # 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import speech_recognition as sr #pip install SpeechRecognition
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
import librosa.display
import librosa


r = sr.Recognizer()


origin_file = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\STT_F_pred\\F_denoise\\1_file\\', ext=['wav'])
volume_file = librosa.util.find_files('C:\\nmb\\nmb_data\\volume\\normal\\F_pred\\', ext=['wav'])
speed_file = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\STT_F_pred\\slow\\', ext=['wav'])

origin = []
for j, path in enumerate(origin_file) : 
    sound_file = AudioSegment.from_wav(path)
    dbfs = sound_file.dBFS
    thresh = int(dbfs)
    if dbfs < thresh :
        thresh = thresh - 1
    audio_chunks = split_on_silence(sound_file,  
        min_silence_len= 200,
        silence_thresh= dbfs - 100,
        keep_silence= 300
    )
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
        origin.append(checked_sent)
print('원본 : ',origin)

volume = []
for j, path in enumerate(volume_file) : 
    sound_file = AudioSegment.from_wav(path)
    dbfs = sound_file.dBFS
    thresh = int(dbfs)
    if dbfs < thresh :
        thresh = thresh - 1
    audio_chunks = split_on_silence(sound_file,  
        min_silence_len= 200,
        silence_thresh= dbfs - 100,
        keep_silence= 300
    )
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
        volume.append(checked_sent)
print('볼륨 : ', volume)

speed = []
for j, path in enumerate(speed_file) : 
    sound_file = AudioSegment.from_wav(path)
    dbfs = sound_file.dBFS
    thresh = int(dbfs)
    if dbfs < thresh :
        thresh = thresh - 1
    audio_chunks = split_on_silence(sound_file,  
        min_silence_len= 200,
        silence_thresh= dbfs - 100,
        keep_silence= 300
    )
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
        speed.append(checked_sent)
print('속도 : ', speed)


# #----------------------------------------------------------------------

# pairs = dict(zip(origin, speed))        # key=value 형태로 dict 를 생성
# lines = map(lambda item:'원본 :{}\n 볼륨 :{}\n'.format(item[0], item[1]), pairs.items())       
# # 코드:{};비용:{}; 형태의 템플릿으로 저장할 문자열 리스트를 생성
# with open('C:\\nmb\\nmb_data\\STT\\test.txt', 'wt') as f: f.writelines(lines) 

pairs = [origin, volume, speed] 

new = ''
for i in range(len(pairs[0])):
    new += '원본 : ' + pairs[0][i] + '\n볼륨 : ' + pairs[1][i] + '\n속도 : ' + pairs[2][i] + '\n\n'

with open('C:\\nmb\\nmb_data\\STT\\test.txt', 'wt') as f: f.writelines(new)        



