from pytube import YouTube
import glob
import os.path

# 먼저 실행 1번
# 유튜브 전용 인스턴스 생성
par = 'https://youtu.be/tWatiCnuK0U'
yt = YouTube(par)
yt.streams.filter()

parent_dir = 'C:\\nmb\\nmb_data\\youtube' # 다운받을 경로
yt.streams.filter().first().download(parent_dir)
print('success')

# 그 다음 실행 2번
import moviepy.editor as mp

clip = mp.VideoFileClip("어느 병원 다녀 소아암 환아를 만난 9살이 하는 말  ODG.mp4")
clip.audio.write_audiofile("C:\\nmb\\nmb_data\\youtube\\audio.wav")