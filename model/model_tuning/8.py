import numpy as np
import os
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,MaxPooling2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.layers import LeakyReLU


# 데이터 불러오기
f_ds = np.load('C:\\nmb\\nmb_data\\npy\\female_denoise_mel_data.npy')
f_lb = np.load('C:\\nmb\\nmb_data\\npy\\female_denoise_mel_label.npy')
m_ds = np.load('C:\\nmb\\nmb_data\\npy\\male_denoise_mel_data.npy')
m_lb = np.load('C:\\nmb\\nmb_data\\npy\\male_denoise_mel_label.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape) 
# (3840, 128, 862) (3840,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=42
)
aaa = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3072, 128, 862, 1) (3072,)
print(x_test.shape, y_test.shape)   # (768, 128, 862, 1) (768,) 

# 모델 구성
model = Sequential()

def block1(x, filters): 
    x = Conv2D(filters,kernel_size=2, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    return MaxPooling2D(pool_size=2, strides=2)(x)

def block2(x, filters): 
    x = Conv2D(filters,kernel_size=3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    return MaxPooling2D(pool_size=2, strides=2)(x)

def block3(x, filters): 
    x = Conv2D(filters,kernel_size=5, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    return MaxPooling2D(pool_size=2, strides=2)(x)

# def block4(x, filters): 
#     x = Conv2D(filters,kernel_size=7, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#     return MaxPooling2D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')
    for i in range(2):
        x = Conv2D(8,kernel_size=5, strides=1, padding='same')(inputs)
        x = LeakyReLU()(x)
        x1 = block1(x, 8)
        x2 = block2(x, 16)
        x3 = block3(x, 32)
        # x4 = block4(x, 64)
        x = Concatenate()([x1,x2,x3])
    x = Concatenate()([x1,x2,x3])
    x = Conv2D(1,kernel_size=(1,1), strides=(1,1), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=1)(x)
    x = Conv2D(5,kernel_size=(5,5), strides=(1,1), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=1)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)
model = build_model(x_train.shape[1:], 2)
print(x_train.shape[1:])    # (128, 862, 1)
model.summary()

model.save('C:/nmb/nmb_data/h5/Conv2D_model_t8.h5')

start = datetime.now()
# 컴파일, 훈련
op = Adam(lr=1e-5)
batch_size =16

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/Conv2D_weight_t8.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])


# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/Conv2D_weight_t8.h5')
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

############################################ PREDICT ####################################
pred = ['C:/nmb/nmb_data/predict/F','C:/nmb/nmb_data/predict/M','C:/nmb/nmb_data/predict/ODD']

count_f = 0
count_m = 0
count_odd = 0

for pred_pathAudio in pred : 
    files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
    files = np.asarray(files)
    for file in files:   
        name = os.path.basename(file)
        length = len(name)
        name = name[0]

        y, sr = librosa.load(file, sr=22050) 
        mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
        pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0 :  # 여성이라고 예측
            print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
            if length > 9 :    # 이상치
                if name == 'F' :
                    count_odd = count_odd + 1                   
            else :
                if name == 'F' :
                    count_f = count_f + 1
                
        else:                   # 남성이라고 예측              
            print(file,'{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))
            if length > 9 :    # 이상치
                if name == 'M' :
                    count_odd = count_odd + 1
            else :
                if name == 'M' :
                    count_m = count_m + 1
                
                    
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")
print("10개 이상치 목소리 중 "+str(count_odd)+"개 정답")


end = datetime.now()
time = end - start
print("작업 시간 : " , time)  


import winsound as sd
def beepsound():
    fr = 440    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

######### 기본 ############
# 1.00E-05
# loss : 0.12706
# acc : 0.95833

# 43개 여성 목소리 중 40개 정답
# 42개 남성 목소리 중 42개 정답
# 10개 이상치 목소리 중 6개 정답
# 0:12:40
##########################

'''
loss : 0.16780
acc : 0.95312
43개 여성 목소리 중 40개 정답
42개 남성 목소리 중 39개 정답
10개 이상치 목소리 중 6개 정답
작업 시간 :  0:00:15.996585
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              [(None, 128, 862, 1) 0
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 128, 862, 8)  208         input[0][0]
__________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)       (None, 128, 862, 8)  0           conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 128, 862, 8)  264         leaky_re_lu_1[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 128, 862, 16) 1168        leaky_re_lu_1[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 128, 862, 32) 6432        leaky_re_lu_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 128, 862, 8)  0           conv2d_5[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 128, 862, 16) 0           conv2d_6[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 128, 862, 32) 0           conv2d_7[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 64, 431, 8)   0           activation_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 64, 431, 16)  0           activation_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 64, 431, 32)  0           activation_5[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 64, 431, 56)  0           max_pooling2d_3[0][0]
                                                                 max_pooling2d_4[0][0]
                                                                 max_pooling2d_5[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 64, 431, 1)   57          concatenate_2[0][0]
__________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)       (None, 64, 431, 1)   0           conv2d_8[0][0]
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 63, 430, 1)   0           leaky_re_lu_2[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 63, 430, 5)   130         max_pooling2d_6[0][0]
__________________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)       (None, 63, 430, 5)   0           conv2d_9[0][0]
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 62, 429, 5)   0           leaky_re_lu_3[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 62, 429, 5)   0           max_pooling2d_7[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 132990)       0           dropout[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          34045696    flatten[0][0]
__________________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)       (None, 256)          0           dense[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 256)          0           leaky_re_lu_4[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            514         dropout_1[0][0]
==================================================================================================
Total params: 34,054,469
Trainable params: 34,054,469
Non-trainable params: 0
__________________________________________________________________________________________________
'''