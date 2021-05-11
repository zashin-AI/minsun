import matplotlib.pyplot as plt
import librosa.display

y_ori, sr_ori = librosa.load('C:\\nmb\\nmb_data\\STT\\P_SUM\\sum.wav')
y_data, sr_data = librosa.load('C:\\nmb\\nmb_data\\STT\\P_DATA\\sum_denoise.wav')
y_reduce, sr_reduce = librosa.load('C:\\nmb\\nmb_data\\STT\\P_REDUCE\\sum_denoise.wav')

fig = plt.figure(figsize = (16, 6))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

librosa.display.waveplot(y_ori, sr = sr_ori, ax = ax1)
librosa.display.waveplot(y_data, sr = sr_data, ax = ax2)
librosa.display.waveplot(y_reduce, sr = sr_reduce, ax = ax3)

fig.tight_layout()
plt.show()