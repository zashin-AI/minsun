
import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
from keras import layers
from speech_vgg import speechVGG
from keras.applications.vgg19 import VGG19


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# 데이터 불러오기

f_ds = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_0_mfccs_40.npy')
f_lb = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_0_mfccs_label_40.npy')
m_ds = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_1_mfccs_40.npy')
m_lb = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_1_mfccs_label_40.npy')
# (1073, 20, 216)
# (1073,)
print('f_lb', f_lb)
print('m_lb', m_lb)

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)
print(y.shape)
# (2141, 20, 862)
# (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape) #(1712, 20, 862)
print(x_test.shape) #(429, 20, 862)
print(y_train.shape) #(1712,)
print(y_test.shape) #(429,)

x_train = x_train.reshape(1712,40,862,1)
x_test = x_test.reshape(429,40,862,1)

model = speechVGG(
            include_top=True,
            input_shape=(40,862,1),
            classes=2,
            pooling=None,
            weights=None,
            transfer_learning=True
        )
model.trainable = False
# model = VGG19(input_shape=(40,431,1), include_top=False, classes=2)
model.summary()

# 컴파일, 훈련
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
mcpath = 'C:/nmb/nmb_data/h5/speechvgg_mfccs_40.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
# history = model.fit(x_train, y_train, epochs=300, batch_size=8, validation_split=0.2, callbacks=[stop, lr, mc])

# --------------------------------------
# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/speechvgg_mfccs_40.h5')

result = model.evaluate(x_test, y_test, batch_size=8)
print('loss: ', result[0]); print('acc: ', result[1])

pred_pathAudio = 'C:/nmb/nmb_data/pred_voice/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    # mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=40, n_fft=512, hop_length=128)
    pred_mfccs = normalize(mfccs, axis=1)
    pred_mfccs = pred_mfccs.reshape(1, 40, 862, 1)
    y_pred = model.predict(pred_mfccs)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')


# =======================================
'''
Model: "speech_vgg"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 40, 862, 1)]      0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 40, 862, 64)       640
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 40, 862, 64)       36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 20, 431, 64)       0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 20, 431, 128)      73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 20, 431, 128)      147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 10, 215, 128)      0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 10, 215, 256)      295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 10, 215, 256)      590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 10, 215, 256)      590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 5, 107, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 5, 107, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 5, 107, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 5, 107, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 2, 53, 512)        0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 2, 53, 512)        2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 2, 53, 512)        2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 2, 53, 512)        2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 1, 26, 512)        0
_________________________________________________________________
flatten_new (Flatten)        (None, 13312)             0
_________________________________________________________________
fc1_new (Dense)              (None, 256)               3408128
_________________________________________________________________
fc2_new (Dense)              (None, 256)               65792
_________________________________________________________________
predictions_new (Dense)      (None, 2)                 514
=================================================================
Total params: 18,187,970
Trainable params: 0
Non-trainable params: 18,187,970
_________________________________________________________________
 '''

# loss:  0.6931481957435608
# acc:  0.5221444964408875
# C:\nmb\nmb_data\pred_voice\FY1.wav 50.004273653030396 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\MZ1.wav 50.00525116920471 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 50.005316734313965 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 50.00661015510559 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 50.00622272491455 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 50.007039308547974 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 50.00602602958679 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 50.00689625740051 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 50.006675720214844 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 50.007081031799316 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 50.006985664367676 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 50.00585317611694 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 50.00694394111633 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 50.00590682029724 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 50.007349252700806 %의 확률로 여자입니다.