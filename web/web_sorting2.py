from flask import Flask, request, render_template, send_file

from tensorflow.keras.models import load_model
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import numpy as np
import librosa
import speech_recognition as sr
import tensorflow as tf
import os
import copy

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

r = sr.Recognizer()

def 
    