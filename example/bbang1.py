import librosa
import librosa.display
import IPython.display # 주피터 노트북에서 쓰면 오디오 파일 실행할 수 있다
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
# 빵형 유튜브

# 1. 오디오 파일 분해하고 그래프 그려서 이해해보기

# audio_path = '../nmb_data/pansori/6rmA6riw7LC9/WRd7fHY3enA/6rmA6riw7LC9-WRd7fHY3enA-0044.flac' # 남자
audio_path = '../nmb_data/pansori/6rmA7Zic7KCV/grgRnDg-o94/6rmA7Zic7KCV-grgRnDg-o94-0001.flac' # 여자(부산)

y, sr = librosa.load(audio_path)
# y = 소리가 떨리는 세기(진폭)를 시간 순서대로 나열한 것 
# sr(sampling rate) = 1초당 샘플의 개수, 단위 Hz(헤르츠) 또는 kHz(킬로헤르츠) (1Hz = 0.001kHz)
# sr이 얼마나 많은가에 따라서 음질이 좋다 안 좋다, 음을 얼마나 많이 표현할 수 있는지, 자세하게 표현할 수 있는지에 대해 결정
# 이미지로 치면 y = 픽셀의 값(RGB), sr = 해상도

# ori_sent = '나 지금 인터넷 뱅킹 시작했어'
ori_sent = '저는 반송이라는 마을에 살고 있는데요'


print(y)
# [0.00694609 0.00433017 0.00230759 ... 0.00080805 0.00091054 0.        ]
print(len(y))
# 75389
print('sr(KHz) : %d' % sr)
# sr(KHz) : 22050
print('오디오 길이(초) : %.2f' % (len(y) / sr))
# 오디오 길이(초) : 3.42

# IPython.display.Audio(data=y, rate=sr)

plt.figure(figsize=(16,6))
librosa.display.waveplot(y=y, sr=sr)
plt.show()
# x축은 시간, y축은 오디오의 세기 / 높게 나와있는 부분은 음이 큰 부분, 낮게 되어있으면 낮은 부분

########################## Fourier Transform ##############################################
# 시간 영역 데이터를 주파수 영역으로 변경
# 좀 더 분석이 쉬워진다
# y축 :  주파수(로그 스케일), color축 : 데시벨(진폭)

D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
# n_fft : 윈도우 사이즈
# hop_length : 
print(D.shape)
# (1025, 148)

plt.figure(figsize=(16,6))
plt.plot(D)
plt.show()


###################### Spectogram #############################
# 시간에 따른 신호 주파수의 스펙트럼 그래프
# 다른 이름 : Sonographs, Voiceprints, Voicegrams

DB = librosa.amplitude_to_db(D, ref=np.max)

plt.figure(figsize=(16,6))
librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()


#################### Mel Spectogram ########################
# (인간이 이해하기 힘든)Spectogram의 y축을 Mel Scale로 변환한 것(Non-linear transformation)
# Mel Scale = log scale
# 가장 많이 이용(이미지로 치면 히스토그램)
S = librosa.feature.melspectrogram(y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(16,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()


##############################################################################################################

# 2. 오디오 특성 추출(Audio Feature Extraction)

# tempo(BPM)
# BPM 정보가 없을 때 librosa의 beat_track 메소드를 통해 BPM을 예측할 수 있다
tempo, _ = librosa.beat.beat_track(y, sr=sr)
print(tempo)


#-----------------------------------------------------------------------------------------------------------------
# Zero Crossing Rate
# 음파가 양에서 음으로 또는 음에서 양으로 바뀌는 비율
zero_crossings = librosa.zero_crossings(y, pad=False)
print(zero_crossings)
print(sum(zero_crossings))
# [False False False ... False False  True]
# 8897

n0 = 5000
n1 = 5040
# 5000에서 5040번대의 시간대를 본다
zero_crossings = librosa.zero_crossings(y[n0:n1], pad=False)
print(sum(zero_crossings)) #3

# zero_crossings :  중간의 0.000 부분을 몇 번 지나쳤나?5000~5040사이는 3번 지나쳤다
plt.figure(figsize=(16,6))
plt.plot(y[n0:n1])
plt.grid()
plt.show()

#-----------------------------------------------------------------------------------------------------------------
# Harmonic and Percussive Components
# Harmonics: 사람의 귀로 구분할 수 없는 특징들 (음악의 색깔)
# Percussives: 리듬과 감정을 나타내는 충격파
# 음악 장르 구분
y_harm, y_perc = librosa.effects.hpss(y)

plt.figure(figsize=(16, 6))
plt.plot(y_harm, color='b')
plt.plot(y_perc, color='r')
plt.show()

#-----------------------------------------------------------------------------------------------------------------
# Spectral Centroid
# 소리를 주파수 표현했을 때, 주파수의 가중평균을 계산하여 소리의 "무게 중심"이 어딘지를 알려주는 지표 
# 예를 들어, 블루스 음악은 무게 중심이 가운데 부분에 놓여있는 반면, 메탈 음악은 (끝 부분에서 달리기 때문에) 노래의 마지막 부분에 무게 중심이 실린다

spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]

# Computing the time variable for visualization
frames = range(len(spectral_centroids))

# Converts frame counts to time (seconds)
t = librosa.frames_to_time(frames)

import sklearn
def normalize(x, axis=0):
  return sklearn.preprocessing.minmax_scale(x, axis=axis)

plt.figure(figsize=(16, 6))
librosa.display.waveplot(y, sr=sr, alpha=0.5, color='b')
plt.plot(t, normalize(spectral_centroids), color='r') # 무게중심 표현
plt.show()

#-----------------------------------------------------------------------------------------------------------------
# Mel-Frequency Cepstral Coefficients (MFCCs)
# MFCCs는 특징들의 작은 집합(약 10-20)으로 스펙트럴 포곡선의 전체적인 모양을 축약하여 보여준다
# 사람의 청각 구조를 반영하여 음성 정보 추출
# https://tech.kakaoenterprise.com/66

mfccs = librosa.feature.mfcc(y, sr=sr)
mfccs = normalize(mfccs, axis=1)

print('mean: %.2f' % mfccs.mean()) # 평균값
print('var: %.2f' % mfccs.var()) # 분산값
# mean: 0.47
# var: 0.05

plt.figure(figsize=(16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()

#-----------------------------------------------------------------------------------------------------------------
# Chroma Frequencies
# chroma는 12개의 반음
# 크로마 특징은 음악의 흥미롭고 강렬한 표현이다
# 크로마는 인간 청각이 옥타브 차이가 나는 주파수를 가진 두 음을 유사음으로 인지한다는 음악이론에 기반한다 = 화음 잘 인식
# 모든 스펙트럼을 12개의 Bin으로 표현한다
# 12개의 Bin은 옥타브에서 12개의 각기 다른 반음(Semitones=Chroma)을 의미한다

chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=512)

plt.figure(figsize=(16, 6))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512)
plt.show()