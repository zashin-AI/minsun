from flask import Flask, request,render_template, send_file
# Flask : 웹 구동을 위한 부분
# request : 파일을 업로드 할 때 flask 서버에서 요청할 때 쓰는 부분
# render_template : html을 불러올 때 필요한 부분
# send_file : 파일을 다운로드 할 때  flask 서버에서 보낼 때 쓰는 부분

from tensorflow.keras.models import load_model
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
from hanspell import spell_checker

import numpy as np
import librosa
import speech_recognition as sr
import tensorflow as tf
import os
import copy

r = sr.Recognizer()

# 볼륨 정규화
def normalized_sound(audio_file):
    audio = AudioSegment.from_wav(audio_file)
    normalizedsound = effects.normalize(audio)
    return normalizedsound

# 말의 앞뒤 묵음 유지한채 묵음 자르기
def split_silence(audio_file):
    dbfs = audio_file.dBFS
    audio_chunks = split_on_silence(
        audio_file,
        min_silence_len=1000,
        silence_thresh=dbfs - 30,
        keep_silence= True
    )
    return audio_chunks

# 네이버 맞춤법 검사기
def STT_hanspell(audio_file):
    with audio_file as audio:
        file = r.record(audio)
        stt = r.recognize_google(file, language= 'ko-KR')
        spelled_sent = spell_checker.check(stt)
        checked_sent = spelled_sent.checked 
    return checked_sent

# 여자 남자 구분 모델
def predict_speaker(y, sr):
    mels = librosa.feature.melspectrogram(y,sr = sr, hop_length=128, n_fft=512, win_length=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max) # 진폭 스펙트로그램을 db스케일 스펙트로그램으로 변환
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    y_pred = model.predcit(pred_mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0:
        return '여자'
    if y_pred_label == 1:
        return '남자'

app = Flask(__name__) 

# 첫 화면 (파일 업로드)
@app.route('/')
def upload_file():
    return render_template('upload.html') # upload.html 파일 템플릿을 가지고 화면 보여준다.

# 업로드 후에 출력 되는 화면
@app.route('/uploadFile', methods = ['POST'])
def download():
    # 파일이 업로드 되면 실시할 과정
    if request.method == 'POST':
        f = request.files['file']
        if not f : return render_template('upload.html')

        # 파일을 받아 볼륨 정규화를 해주고
        normalizedsound = normalized_sound(f)
        # 묵음을 잘라준다
        audio_chunks = split_silence(normalizedsound)

        save_script = ''

        for i, chunk in enumerate(audio_chunks):
            speaker_stt = list()
            out_file = "chunk.wav"
            chunk.export(out_file, format = 'wav')
            aaa = sr.AudioFile(out_file)

            try :
                stt_text = STT_hanspell(aaa)
                speaker_stt.append(str(stt_text))

                y, sample_rate = librosa.load(out_file, sr = 22050)

                #만약 음성 파일의 길이가 5초 이상이거나 같다면
                if len(y) >= 22050*5:
                    y = y[:22050*5]
                    speaker = predict_speaker(y, sample_rate)
                    speaker_stt.append(str(speaker))
                    print(speaker_stt[1], ":", speaker_stt[0])
                
                # 음성 파일의 길이가 5초 미만이라면
                else : 
                    audio_copy = AudioSegment.from_wav(out_file)
                    audio_copy = copy.deepcopy(audio_copy) # deepcopy = 내부에 객체들까지 모두 새롭게 copy된다.
                    for num in range(3):
                        audio_copy = audio_copy.append(copy.deepcopy(audio_copy), crossfade=0) 
                        # crossfade를 0으로 해서 append 메서드를 호출한다
                        # sum(), reduce() 등을 사용하여 총 지속 시간을 변경하지 않고 여러 chunk를 다시 결합할 수 있도록 한다.
                    out_file_over5s = "chunk_over_5s.wav"
                    audio_copy.export(out_file_over5s, format = 'wav')
                    y_copy, sample_rate = librosa.load(out_file_over5s, sr = 22050)
                    y_copy = y_copy[:22050*5]
                    speaker = predict_speaker(y_copy, sample_rate)
                    speaker_stt.append(str(speaker))
                    print(speaker_stt[1], ":", speaker_stt[0])

                save_script += speaker_stt[1] + ":" + speaker_stt[0] + '\n\n'
                with open('C:/nmb/nada/web/static/test.txt', 'wt', encoding='utf-8') as f : f.writelines(save_script)

                # chunk.wav파일 삭제하기
                if os.path.isfile(out_file):
                    os.remove(out_file)

                if os.path.isfile(out_file_over5s):
                    os.remove(out_file_over5s)
            except:
                pass # 너무 짧은 음성은 패스하도록 한다.
        return render_template('/download.html')

# 파일 다운로드
@app.route('/download/')
def download_file():
    file_name = 'C:/nmb/nada/web/static/test.txt'
    return send_file(
        file_name,
        as_attachment=True, # False의 경우 파일로 다운로드가 안되고 화면에 출력된다.
        mimetype='text/txt',
        cache_timeout=0 # 지정한 파일이 아니라 과거의 파일이 계속 다운 받아지는 경우, 캐시메모리의 타임아웃을 0으로 지정해주면 된다.
        )

# 추론 된 파일 읽기
@app.route('/read')
def read_text():
    f = open('C:/nmb/nada/web/static/test.txt', 'r', encoding='utf-8')
    return "</br>".join(f.readlines())

if __name__ == '__main__' :
    model = load_model('C:\\nmb\\nmb_data\\h5\\mobilenet_rmsprop_1.h5')
    app.run(debug=True) # debug = False 인 경우 문제가 생겼을 경우 제대로 된 확인을 하기 어려움






