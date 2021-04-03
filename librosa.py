import librosa
import librosa.display
import IPython.display # 주피터 노트북에서 쓰면 오디오 파일 실행할 수 있다
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

stft_result = librosa.stft(y, n_fft=4096, win_length = 4096, hop_length=512)
D = np.abs(stft_result)
S_dB = librosa.power_to_db(D, ref=np.max)
librosa.display.specshow(S_dB, sr=sr, hop_length = 1024, y_axis='mel', x_axis='time', cmap = cm.jet)
plt.colorbar(format='%2.0f dB')
plt.show()


# y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
# librosa.feature.chroma_stft(y=y, sr=sr)
# array([[1.   , 0.962, ..., 0.143, 0.278],
#        [0.688, 0.745, ..., 0.103, 0.162],
#        ...,
#        [0.468, 0.598, ..., 0.18 , 0.342],
#        [0.681, 0.702, ..., 0.553, 1.   ]], dtype=float32)


# S = np.abs(librosa.stft(y))
# chroma = librosa.feature.chroma_stft(S=S, sr=sr)
# print(chroma)
# array([[1.   , 0.973, ..., 0.527, 0.569],
#        [0.774, 0.81 , ..., 0.518, 0.506],
#        ...,
#        [0.624, 0.73 , ..., 0.611, 0.644],
#        [0.766, 0.822, ..., 0.92 , 1.   ]], dtype=float32)

# S = np.abs(librosa.stft(y, n_fft=4096))**2
# chroma = librosa.feature.chroma_stft(S=S, sr=sr)
# print(chroma)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
# fig.colorbar(img, ax=ax)
# ax.set(title='Chromagram')