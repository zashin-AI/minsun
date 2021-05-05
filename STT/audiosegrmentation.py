# https://github.com/jiaaro/pydub/issues/169

from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence      # 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import speech_recognition as sr #pip install SpeechRecognition
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
import librosa.display
import librosa


r = sr.Recognizer()

'''디노이즈 > 속도 느리게 > 오디오 파일을 불러옴 > silence 부분마다 잘라서 음성 파일 저장 > 해당 파일을 google stt에 적용 > 한글 맞춤법 검사 > text 출력'''

# 원래 STT 파일
# file_list = librosa.util.find_files('E:\\nmb\\nmb_data\\predict\\stt', ext=['wav'])

# 디노이즈 적용
# file_list = librosa.util.find_files('E:\\nmb\\nmb_data\\predict\\stt_denoise\\denoise', ext=['wav'])

# slow
file_list = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\mindslab\\split2m\\1\\', ext=['wav'])
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
        min_silence_len= 200,
        silence_thresh= dbfs - 16 ,
        keep_silence= 100
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
            full_txt.append(str(checked_sent))
            # print(txt)
            # print(full_txt)
        except : # 너무 짧은 음성은 pass 됨 
            pass
    print(path , '\n', full_txt)

    with open('C:\\nmb\\nmb_data\\STT\\mindslab_1.txt', 'wt') as f: f.writelines(checked_sent)        

'''   
C:\nmb\nmb_data\STT\STT voice denoise\test_01_denoise.wav
 ['하루 확진자가 오늘로 나흘째 600명 때에 머물고 있습니다', '하지만 이스라엘은 사실상 집단면역 선언', '오늘부터 야외에서 마스크를 벗고 있습니다 JTBC 취재팀이 직접 이스라엘로 날아갔는데 잠시 후 상지 연
결해 보겠습니다']
C:\nmb\nmb_data\STT\STT voice denoise\test_02_denoise.wav
 ['토끼와 자라', '옛날에', '어느 바닷속에 아주 아름다운 용궁이 있었어요', '그런데', '이 아름다운 용궁에 슬픈 일이 생겼답니다', '나이 많은 용왕님이', '시름시름 앓다가 자리에 누워 있기 때문이지요']     
C:\nmb\nmb_data\STT\STT voice denoise\test_F1_denoise.wav
 ['사람 간이 m 이상 거리 2', '거리두기 지침에 따라 이 자리는 비워 주소']
C:\nmb\nmb_data\STT\STT voice denoise\test_F2_denoise.wav
 ['사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요']
C:\nmb\nmb_data\STT\STT voice denoise\test_F3_denoise.wav
 ['사람 간이 m 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요']
C:\nmb\nmb_data\STT\STT voice denoise\test_F4_denoise.wav
 ['실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비어 주세요']
C:\nmb\nmb_data\STT\STT voice denoise\test_M1_denoise.wav
 ['실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요']
C:\nmb\nmb_data\STT\STT voice denoise\test_M2_denoise.wav
 ['실내 가람과 2미터 이상 거리', '기침 안 따라 이 자리는 비워 줘']
C:\nmb\nmb_data\STT\STT voice denoise\test_M3_denoise.wav
 ['실내 사람 간이 m 이상 거리두기', '거리두기 지침에 따라 이 자리는 비워 주세요']
C:\nmb\nmb_data\STT\STT voice denoise\test_M4_denoise.wav
 ['실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비어 주소']
C:\nmb\nmb_data\STT\STT voice denoise\test_M5_denoise.wav
 ['실내 사람 간이 m 이상 거리', '거리두기 지침 했다', '이 자리는 비워 주세요']
C:\nmb\nmb_data\STT\STT voice denoise\test_M6_denoise.wav
 ['실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 줘']
 '''