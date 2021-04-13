from pytube import YouTube
import glob
import os.path

# 먼저 실행 1번
# 유튜브 전용 인스턴스 생성
par = 'https://youtu.be/ytZorAGWqmQ'
yt = YouTube(par)
yt.streams.filter()

parent_dir = 'C:\\nmb\\nmb_data\\youtube' # 다운받을 경로
yt.streams.filter().first().download(parent_dir)
print('success')

# 그 다음 실행 2번
import moviepy.editor as mp

clip = mp.VideoFileClip("남녀노소 쳐발리는 걸그룹 중저음탁성허스키 보이스 멤버 모음zip (이서연 장규리 조유리 송우기 전희진).mp4")
clip.audio.write_audiofile("C:\\nmb\\nmb_data\\youtube\\audio2.wav")