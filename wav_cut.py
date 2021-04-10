from pydub import AudioSegment
import os

print(os.getcwd())  

if not os.path.isdir("splitaudio1(소아암)"):
    os.mkdir("splitaudio1(소아암)")

audio = AudioSegment.from_file("C:\\nmb\\nmb_data\\youtube\\audio.wav")
lengthaudio = len(audio)
print("Length of Audio File", lengthaudio)

start = 0
# In Milliseconds, this will cut 5 Sec of audio
threshold = 5000
end = 0
counter = 0

while start < len(audio):
    end += threshold
    print(start , end)
    chunk = audio[start:end]
    filename = f"C:\\nmb\\nmb_data\\youtube\\chunk{counter}.wav"
    chunk.export(filename, format="wav")
    counter +=1
    start += threshold


# from pydub import AudioSegment
# from pydub.silence import split_on_silence

# sound = AudioSegment.from_wav("C:\\nmb\\nmb_data\\youtube\\audio.wav")
# chunks = split_on_silence(sound, 
#     # must be silent for at least half a second
#     min_silence_len=5000,

#     # consider it silent if quieter than -16 dBFS
#     silence_thresh=-16
# )

# for i, chunk in enumerate(chunks):
#     chunk.export("C:\\nmb\\nmb_data\\youtube\\chunk{0}.wav".format(i), format="wav")

#     #   warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)

