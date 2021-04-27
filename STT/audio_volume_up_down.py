# https://www.javaer101.com/ko/article/21996994.html

from pydub import AudioSegment,effects
import librosa, librosa.display
import matplotlib.pyplot as plt

# 파일 생성 안됨
'''
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

sound = AudioSegment.from_file("C:\\nmb\\nmb_data\\volume\\denoise\\friendvoice_F4_denoise.wav", "wav")
normalized_sound = match_target_amplitude(sound, -20.0)
normalized_sound.export("C:\\nmb\\nmb_data\\volume\\normal\\friendvoice_F4_normal.wav", format="wav")
'''
########################################################################################

path = "C:\\nmb\\nmb_data\\volume\\denoise\\"
outpath = "C:\\nmb\\nmb_data\\volume\\normal\\"

file = path + "F (12)_0_denoise.wav"

song = AudioSegment.from_wav(file)

# +-db
song = song + 15

#save the output
song.export(outpath+"F (12)_0_up.wav", "wav")

y1, sr1 = librosa.load('C:\\nmb\\nmb_data\\volume\\denoise\\F (12)_0_denoise.wav')
y2, sr2 = librosa.load('C:\\nmb\\nmb_data\\volume\\normal\\F (12)_0_up.wav')

plt.figure(figsize=(16, 10))
plt.subplot(4,1,1)
librosa.display.waveplot(y=y1, sr=sr1)
plt.title('denoise')

plt.subplot(4,1,3)
librosa.display.waveplot(y=y2, sr=sr2)
plt.title('down')

plt.show()
