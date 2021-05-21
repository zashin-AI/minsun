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
    audio_chunck = split_on_silence(
        audio_file,
        min_silence_len=1000,
        silence_thresh=dbfs - 30,
        keep_silence= True
    )
    return audio_chunck



