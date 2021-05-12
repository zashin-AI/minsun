
from keras import optimizers
import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard
from keras.models import Model
from keras import layers
from speech_vgg import speechVGG
from sklearn.metrics import accuracy_score, recall_score, precision_score
from datetime import datetime
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Nadam
from keras import backend as K


# 데이터 불러오기

x = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_data.npy') # data
y = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_label.npy') # label
# (1073, 20, 216)
# (1073,)
x = x.reshape(-1, x.shape[1] * x.shape[2])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=42)

print(x.shape, y.shape) # (4536, 128, 862) (4536,)
print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862) (908,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(3628, 128, 862,1)
x_test = x_test.reshape(908, 128, 862,1)

model = speechVGG(
    include_top=True,
    input_shape=(128,862,1),
    classes=2,
    pooling=None,
    weights=None,
    transfer_learning=True
)

model.summary()
# model.trainable = False

start = datetime.now()
'''
############################################## f1 score ####################33##################################
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score
###############################################################################################################

# 컴파일, 훈련
op=Adadelta(lr=1e-5)
# model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=["acc",f1score])
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=["acc"])
stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
mcpath = 'C:/nmb/nmb_data/h5/speechvgg_mels_del4.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=5000, batch_size=8, validation_split=0.2, callbacks=[stop, lr, mc, tb])

# --------------------------------------
# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/speechvgg_mels_del4.h5')

# _loss, _acc, _f1score = model.evaluate(x_test, y_test, batch_size=8)
# print('loss: {:.4f}, accuracy: {:.4f}, f1score: {:.4f}'.format(_loss, _acc, _f1score))

_loss, _acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss: {:.4f}, accuracy: {:.4f}'.format(_loss, _acc))

# result = model.evaluate(x_test, y_test, batch_size=8)
# print('loss: ', result[0]); print('acc: ', result[1])

# y_pred = model.predict(x_test)

# y_pred_label = np.argmax(y_pred)

# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# print("accuracy : \t", accuracy)
# print("recall : \t", recall)
# print("precision : \t", precision)

pred_pathAudio = 'C:/nmb/nmb_data/pred_voice/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
    mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = mels.reshape(1, 128, 862, 1)
    y_pred = model.predict(pred_mels)
    y_pred_label = np.argmax(y_pred)
   
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')


end = datetime.now()
time = end - start
print("작업 시간 : " , time)  

'''
'''
Model: "speech_vgg"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 128, 862, 1)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 128, 862, 64)      640
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 128, 862, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 64, 431, 64)       0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 64, 431, 128)      73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 64, 431, 128)      147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 32, 215, 128)      0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 32, 215, 256)      295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 32, 215, 256)      590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 32, 215, 256)      590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 16, 107, 256)      0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 16, 107, 512)      1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 16, 107, 512)      2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 16, 107, 512)      2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 8, 53, 512)        0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 8, 53, 512)        2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 8, 53, 512)        2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 8, 53, 512)        2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 26, 512)        0
_________________________________________________________________
flatten_new (Flatten)        (None, 53248)             0
_________________________________________________________________
fc1_new (Dense)              (None, 256)               13631744
_________________________________________________________________
fc2_new (Dense)              (None, 256)               65792
_________________________________________________________________
predictions_new (Dense)      (None, 2)                 514
=================================================================
Total params: 28,411,586
Trainable params: 28,411,586
Non-trainable params: 0
_________________________________________________________________

'''

'''
loss: 0.2115, accuracy: 0.9347, f1score: 0.6150
C:\nmb\nmb_data\pred_voice\FY1.wav 90.99363088607788 %의 확률로 여자입니다.
C:\nmb\nmb_data\pred_voice\MZ1.wav 91.18536114692688 %의 확률로 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 98.60531091690063 %의 확률로 여자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 93.47882270812988 %의 확률로 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 94.38238143920898 %의 확률로 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 93.61133575439453 %의 확률로 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 95.84023356437683 %의 확률로 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 95.73729634284973 %의 확률로 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 96.66086435317993 %의 확률로 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 82.44102001190186 %의 확률로 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 98.047536611557 %의 확률로 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 93.25112104415894 %의 확률로 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 92.67841577529907 %의 확률로 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 67.82888174057007 %의 확률로 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 95.23817300796509 %의 확률로 남자입니다.
작업 시간 :  0:00:16.206817
'''