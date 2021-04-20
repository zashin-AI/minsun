
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

f_ds = np.load('C:\\nmb\\nmb_data\\npy\\M100_0_mels.npy')
f_lb = np.load('C:\\nmb\\nmb_data\\npy\\M100_0_mels_label.npy')
m_ds = np.load('C:\\nmb\\nmb_data\\npy\\M100_1_mels.npy')
m_lb = np.load('C:\\nmb\\nmb_data\\npy\\M100_1_mels_label.npy')
# (1073, 20, 216)
# (1073,)
print('f_lb', f_lb)
print('m_lb', m_lb)

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)
print(y.shape)
# (1173, 128, 862)
# (1173,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape) #(938, 128, 862)
print(x_test.shape) #(235, 128, 862)
print(y_train.shape) #(1712,)
print(y_test.shape) #(429,)

x_train = x_train.reshape(x_train.shape[0],128,862,1)
x_test = x_test.reshape(x_test.shape[0],128,862,1)
'''
# 베스트 파라미터 찾기
def modeling(optimizer='adam', learning_rate=0.1):
    model = speechVGG(
                include_top=True,
                input_shape=(128,862,1),
                classes=2,
                pooling=None,
                weights=None,
                transfer_learning=True
            )

    model.summary()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])
    return model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=modeling, verbose=1)
parameters = {'optimizer' :  ['adam','sgd', 'adadelta','rmsprop'],
              'learning_rate' : [0.1,0.01,0.001,0.0001,0.00001]}
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model2,parameters,cv=3)
search = GridSearchCV(model2,parameters,cv=3)

# search.fit(x,y)
# {'optimizer': 'sgd', 'learning_rate': 0.001}
search.fit(x_train,y_train)
# {'learning_rate': 1e-05, 'optimizer': 'adadelta'}

print(search.best_params_) # 내가 선택한 파라미터 중 제일 좋은 것
print(search.best_estimator_) # 전체 추정기 중에서 가장 좋은 것
'''

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

# 컴파일, 훈련
op=Adadelta(lr=1e-5)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=["acc", f1score])
stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
mcpath = 'C:/nmb/nmb_data/h5/speechvgg_mels_del4.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
# history = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[stop, lr, mc, tb])

# --------------------------------------
# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/speechvgg_mels_del4.h5')


_loss, _acc, _f1score = model.evaluate(x_test, y_test, batch_size=8)
print('loss: {:.3f}, accuracy: {:.3f}, f1score: {:.3f}'.format(_loss, _acc, _f1score))


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
loss: 0.185, accuracy: 0.945, f1score: 0.080
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
작업 시간 :  0:00:15.465837
'''