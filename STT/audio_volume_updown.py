from pydub import AudioSegment, effects
import os,librosa, librosa.display
import sys
sys.path.append('c:/nmb/nada/python_import/')
from volume_handling import volume_updown
import matplotlib.pyplot as plt

# path = "C:\\nmb\\nmb_data\\STT\\STT voice denoise\\test_M5_denoise.wav"
# outpath = "C:\\nmb\\nmb_data\\volume\\updown\\"
# volume = -10

# volume_updown(path=path, volume=volume, outpath=outpath)


y1, sr1 = librosa.load('C:\\nmb\\nmb_data\\volume\\normal\\F_pred\\130_003_0052_denoise_volume_normal.wav')
y2, sr2 = librosa.load('C:\\nmb\\nmb_data\\volume\\updown\\F_pred\\130_003_0052_denoise+30db_up.wav')

plt.figure(figsize=(16, 20))
plt.subplot(4,1,1)
librosa.display.waveplot(y=y1, sr=sr1)
plt.title('denoise')

plt.subplot(4,1,3)
librosa.display.waveplot(y=y2, sr=sr2)
plt.title('updown')

plt.show()