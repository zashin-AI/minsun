# https://www.javaer101.com/ko/article/21996994.html

from pydub import AudioSegment,effects
import librosa, librosa.display
import matplotlib.pyplot as plt

# def match_target_amplitude(sound, target_dBFS):
#     change_in_dBFS = target_dBFS - sound.dBFS
#     return sound.apply_gain(change_in_dBFS)

# sound = AudioSegment.from_file("C:\\nmb\\nmb_data\\volume\\denoise\\", "wav")
# normalized_sound = match_target_amplitude(sound, -20.0)
# normalized_sound.export("C:\\nmb\\nmb_data\\volume\\up\\nomrmalizedAudio.wav", format="wav")

# song = AudioSegment.from_wav('C:\\nmb\\nmb_data\\volume\\denoise\\F (12)_0_denoise.wav')

# # reduce volume by 10 dB
# song_10_db_quieter = song - 10

# # but let's make him *very* quiet
# song = song - 36

# # save the output
# song.export("C:\\nmb\\nmb_data\\volume\\down\\F (12)", "wav")



from pydub import AudioSegment, effects  

rawsound = AudioSegment.from_file("C:\\nmb\\nmb_data\\volume\\denoise\\friendvoice_F4_denoise.wav", "wav")  
normalizedsound = effects.normalize(rawsound)  
normalizedsound.export("C:\\nmb\\nmb_data\\volume\\normal\\friendvoice_F4_normal.wav", format="wav")


y1, sr1 = librosa.load('C:\\nmb\\nmb_data\\volume\\denoise\\friendvoice_F4_denoise.wav')
y2, sr2 = librosa.load('C:\\nmb\\nmb_data\\volume\\normal\\friendvoice_F4_normal.wav')

plt.figure(figsize=(16, 6))
plt.subplot(4,1,1)
librosa.display.waveplot(y=y1, sr=sr1)
plt.title('denoise')

plt.subplot(4,1,3)
librosa.display.waveplot(y=y2, sr=sr2)
plt.title('nomal')

plt.show()
