
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

# 데이터 불러오기

f_ds = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_0_mels.npy')
f_lb = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_0_mels_label.npy')
m_ds = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_1_mels.npy')
m_lb = np.load('C:\\nmb\\nmb_data\\npy\\brandnew_1_mels_label.npy')
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
print(x_train.shape) #(1712, 128, 862)
print(x_test.shape) #(429, 128, 862)
print(y_train.shape) #(1712,)
print(y_test.shape) #(429,)

x_train = x_train.reshape(1712,128,862,1)
x_test = x_test.reshape(429,128,862,1)
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


start = datetime.now()

# 컴파일, 훈련
op=RMSprop(lr=1e-4)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=["acc"])
stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
mcpath = 'C:/nmb/nmb_data/h5/speechvgg_mels_rms3.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=8, validation_split=0.2, callbacks=[stop, lr, mc, tb])

# --------------------------------------
# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/speechvgg_mels_rms3.h5')

result = model.evaluate(x_test, y_test, batch_size=8)
print('loss: ', result[0]); print('acc: ', result[1])

y_pred = model.predict(x_test)

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
