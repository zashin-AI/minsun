# https://towardsdatascience.com/audio-classification-with-pre-trained-vgg-19-keras-bca55c2a0efe


# Download Audio Files from Youtube

from __future__ import unicode_literals
import youtube_dl


ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',.
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([<youtube video link>])
# for bike sounds : https://www.youtube.com/watch?v=sRdRwHPjJPk
# for car sounds : https://www.youtube.com/watch?v=PPdNb-XQXR8

# Converting Audio Files from .mp3 to .wav

from pydub import AudioSegment
sound = AudioSegment.from_mp3("car.mp3")
sound.export("car.wav", format="wav")

# Extracting Chunks of Audios
from pydub import AudioSegment
import os
if not os.path.exists("bike"):
    os.makedirs("bike")

count=1
for i in range(1,1000,15):
    t1 = i * 1000 #Works in milliseconds
    t2 = (i+15) * 1000
    newAudio = AudioSegment.from_wav("bikes.wav")
    newAudio = newAudio[t1:t2]
    newAudio.export('bike/'+str(count)+'.wav', format="wav") #Exports to a wav file in the current path.
    print(count)
    count+=1

# Plotting Amplitude Waveforms
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from os import walk
import os
if not os.path.exists("carPlots"):
    os.makedirs("carPlots")
car_wavs = []
for (_,_,filenames) in walk('car'):
    car_wavs.extend(filenames)
    break
for car_wav in car_wavs:
    # read audio samples
    input_data = read("car/" + car_wav)
    audio = input_data[1]
    # plot the first 1024 samples
    plt.plot(audio)
    # label the axes
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    # set the title
    # plt.title("Sample Wav")
    # display the plot
    plt.savefig("carPlots/" + car_wav.split('.')[0] + '.png')
    # plt.show()
    plt.close('all')

# Extracting Features and Training LinearSVM
import os
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

def get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    return list(flatten[0])

X = []
y = []

car_plots = []
for (_,_,filenames) in os.walk('carPlots'):
    car_plots.extend(filenames)
    break

for cplot in car_plots:
    X.append(get_features('carPlots/' + cplot))
    y.append(0)

bike_plots = []
for (_,_,filenames) in os.walk('bikePlots'):
    bike_plots.extend(filenames)
    break

for cplot in bike_plots:
    X.append(get_features('bikePlots/' + cplot))
    y.append(1)

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

# get the accuracy
print (accuracy_score(y_test, predicted))