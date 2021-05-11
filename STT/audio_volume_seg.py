from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence      # 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import speech_recognition as sr #pip install SpeechRecognition
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
import librosa.display
import librosa


r = sr.Recognizer()

file_list = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\corpus\\noise\\', ext=['wav'])

full_txt =[]
for j, path in enumerate(file_list) : 

    sound_file = AudioSegment.from_wav(path)
    dbfs = sound_file.dBFS
    thresh = int(dbfs)
    if dbfs < thresh :
        thresh = thresh - 1
    audio_chunks = split_on_silence(sound_file,  
        # 변경
        min_silence_len= 1000,
        silence_thresh= dbfs - 16,
        keep_silence= 500
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
            full_txt.append(str(checked_sent)) 
        except : # 너무 짧은 음성은 pass 됨 
            pass
    print("파일 이름 : ",path[path.rfind('\\') + 1:])
    # rfind를 사용하여 오른쪽부터 '\\'의 인덱스를 찾고, +1부터 문자열의 마지막까지 가져온다.
    for checked_sent in full_txt:
        print(checked_sent)

with open('C:\\nmb\\nmb_data\\STT\\noise_1000_16_500.txt', 'wt') as f: f.writelines(full_txt)        


# 원본
'''
F (12)_0.wav : 이들은 김 위원장이 나타나자마자 박수와 함께 김종인과 박영선등
friendvoice_F4 : 실내 사람간 2m 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요
testF (5)_0 : 보유 중인 부동산 자산을 추가로 매각해 총 1조 4천억
testF (6)_0 : 천 구백 육십 년대 초반 우리는 남부 아프리카에서 여행하는 봉사를
'''

# original
'''
C:\nmb\nmb_data\volume\original\F (12)_0.wav 
 ['맞아 맞아 김종인과 박영선']
C:\nmb\nmb_data\volume\original\friendvoice_F4.wav 
 ['실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비어 주세요']
C:\nmb\nmb_data\volume\original\testF (5)_0.wav 
 ['보유 중인 부동산 자산을 추가로 매각해 총 1조 4천억']
C:\nmb\nmb_data\volume\original\testF (6)_0.wav 
 ['1960년대 초반 우리는 남부 아프리카에서 여행하는 봉사를']
'''

# 디노이즈
'''
C:\nmb\nmb_data\volume\denoise\F (12)_0_denoise.wav 
 ['원장이 나타나자마자 박도 함께 김종인과 박영선']
C:\nmb\nmb_data\volume\denoise\friendvoice_F4_denoise.wav 
 ['실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비어 주세요']
C:\nmb\nmb_data\volume\denoise\testF (5)_0_denoise.wav 
 ['보유 중인 부동산 자산을 추가로 매각해 총 1조 4천억']
C:\nmb\nmb_data\volume\denoise\testF (6)_0_denoise.wav 
 ['1960년대 초반 우리는 남부 아프리카에서 여행하는 봉사를']
'''

# F (12)_0.wav : 이들은 김 위원장이 나타나자마자 박수와 함께 김종인과 박영선등

# normal
# C:\nmb\nmb_data\volume\up\F (12)_0_up.wav 
#  ['이들은 김 위원장이 나타나자마자 박수 함께 김종인과 박영선']

# up +50 -> 인식을 못한다
# C:\nmb\nmb_data\volume\normal\F (12)_0_up.wav 
#  []

# up + 30
# C:\nmb\nmb_data\volume\normal\F (12)_0_up.wav 
#  ['이들은 김 위원장이 나타나자마자 박수와 함께 김종인과 박영선']

# up +15
# C:\nmb\nmb_data\volume\normal\F (12)_0_up.wav 
#  ['이들은 김 위원장이 나타나자마자 박수 함께 김종인과 박영선']

# _____________________________________________________________________________
# friendvoice_F4 : 실내 사람간 2m 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요

# down -20
# C:\nmb\nmb_data\volume\normal\friendvoice_F4_down.wav 
#  ['사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요']

# 원본
# C:\nmb\nmb_data\volume\denoise\friendvoice_F4_denoise.wav 
#  ['실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비어 주세요']

# up + 20
#  C:\nmb\nmb_data\volume\normal\friendvoice_F4_up.wav 
#  ['실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요']