from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence      # 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import speech_recognition as sr #pip install SpeechRecognition
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
import librosa.display
import librosa


r = sr.Recognizer()

# file_list = librosa.util.find_files('C:\\nmb\\nmb_data\\volume\\normal\\F_pred\\', ext=['wav'])
# print(file_list)
file_list = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\STT voice\\F_pred', ext=['wav'])

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

        # 기본
        min_silence_len= 200,
        silence_thresh= dbfs - 16,
        keep_silence= 100

        # 변경
        # min_silence_len= 200,
        # silence_thresh= dbfs - 100,
        # keep_silence= 300
    )
    # print(len(audio_chunks))

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
            full_txt.append(str(checked_sent)) # 하나로 합칠 경우 사용
            # print(txt)
            # print(full_txt)
        except : # 너무 짧은 음성은 pass 됨 
            pass
    print("파일 이름 : ",path[path.rfind('\\') + 1:])
    # rfind를 사용하여 오른쪽부터 '\\'의 인덱스를 찾고, +1부터 문자열의 마지막까지 가져온다.
    for checked_sent in full_txt:
        print(checked_sent)
    # print('\n')

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