import numpy as np
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop


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
# Convolution - Batch Normalization - Activation - Dropout - Pooling

model = Sequential()


activation = 'relu'
inputs = Input(shape=x_train.shape[1:])
x = Conv2D(16,3, padding='same')(inputs)
x = Activation(activation)(x)
x = Conv2D(16, 3, padding='same')(x)
x = Activation(activation)(x)
s = Conv2D(16, 1,padding='same')(inputs)
x = Add()([x,s])
x = Activation('relu')(x)
x = MaxPooling2D(2, 1)(x)

x = Conv2D(32, 3, padding='same')(x)
x = Activation(activation)(x)
x = Conv2D(32, 3, padding='same')(x)
x = Activation(activation)(x)
s = Conv2D(32, 1,padding='same')(inputs)
x = Add()([x,s])
x = Activation('relu')(x)
x = MaxPooling2D(2, 1)(x)

x = Conv2D(64,3, padding='same')(x)
x = Activation(activation)(x)
x = Conv2D(64, 3, padding='same')(x)
x = Activation(activation)(x)
s = Conv2D(64, 1,padding='same')(inputs)
x = Add()([x,s])
x = Activation('relu')(x)
x = MaxPooling2D(2, 1)(x)

x = Conv2D(128,3, padding='same')(x)
x = Activation(activation)(x)
x = Conv2D(128, 3, padding='same')(x)
x = Activation(activation)(x)
s = Conv2D(128, 1,padding='same')(inputs)
x = Add()([x,s])
x = Activation('relu')(x)
x = MaxPooling2D(2, 2)(x)

x = Conv2D(256,3, padding='same')(x)
x = Activation(activation)(x)
x = Conv2D(256, 3, padding='same')(x)
x = Activation(activation)(x)
s = Conv2D(256, 1,padding='same')(inputs)
x = Add()([x,s])
x = Activation('relu')(x)
x = MaxPooling2D(2, 2)(x)

x = AveragePooling2D(pool_size=3, strides=3)(x)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs , outputs=outputs)
model.summary()

model.save('C:/nmb/nmb_data/h5/Conv2D_my_model_Adam.h5')

# start = datetime.now()
# # 컴파일, 훈련
# op = Adam(lr=1e-5)
# batch_size = 64

# es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
# lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
# path = 'C:/nmb/nmb_data/h5/Conv2D_my_weight_Adam_5.h5'
# mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
# tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
# model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])


# # 평가, 예측
# model.load_weights('C:/nmb/nmb_data/h5/Conv2D_my_weight_Adam_5.h5')
# result = model.evaluate(x_test, y_test, batch_size=batch_size)
# print("loss : {:.5f}".format(result[0]))
# print("acc : {:.5f}".format(result[1]))

# pred_pathAudio = 'C:/nmb/nmb_data/predict/F'
# files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
# files = np.asarray(files)
# count_f = 0

# for file in files:   
#     y, sr = librosa.load(file, sr=22050) 
#     mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
#     pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
#     pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
#     y_pred = model.predict(pred_mels)
#     y_pred_label = np.argmax(y_pred)
#     if y_pred_label == 0 :
#         count_f += 1                 
#         print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
#     else:                         
#         print(file, '{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))

# pred_pathAudio = 'C:/nmb/nmb_data/predict/M'
# files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
# files = np.asarray(files)
# count_m = 0
# for file in files:   
#     y, sr = librosa.load(file, sr=22050) 
#     mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
#     pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
#     pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
#     y_pred = model.predict(pred_mels)
#     y_pred_label = np.argmax(y_pred)
#     if y_pred_label == 0 :               
#         print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
#     else:
#         count_m += 1                              
#         print(file, '{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))

# print("47개 여성 목소리 중 "+str(count_f)+"개 정답")
# print("48개 남성 목소리 중 "+str(count_m)+"개 정답")


# end = datetime.now()
# time = end - start
# print("작업 시간 : " , time)  


# # import winsound as sd
# # def beepsound():
# #     fr = 800    # range : 37 ~ 32767
# #     du = 500     # 1000 ms ==1second
# #     sd.Beep(fr, du) # winsound.Beep(frequency, duration)

# # beepsound()

'''
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 128, 862, 1) 0
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 128, 862, 16) 160         input_1[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 128, 862, 16) 0           conv2d[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 128, 862, 16) 2320        activation[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 862, 16) 0           conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 128, 862, 16) 272         activation_1[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 128, 862, 16) 0           activation_1[0][0]
                                                                 conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 128, 862, 16) 0           add[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 64, 431, 16)  0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 431, 32)  4640        max_pooling2d[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 64, 431, 32)  0           conv2d_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 64, 431, 32)  9248        activation_3[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 64, 431, 32)  0           conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 64, 431, 32)  1056        activation_4[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 64, 431, 32)  0           activation_4[0][0]
                                                                 conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 64, 431, 32)  0           add_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 32, 215, 32)  0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 215, 64)  18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 215, 64)  0           conv2d_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 215, 64)  36928       activation_6[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 215, 64)  0           conv2d_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 215, 64)  4160        activation_7[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 215, 64)  0           activation_7[0][0]
                                                                 conv2d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 215, 64)  0           add_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 16, 107, 64)  0           activation_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 107, 128) 73856       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 16, 107, 128) 0           conv2d_9[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 107, 128) 147584      activation_9[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 16, 107, 128) 0           conv2d_10[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 107, 128) 16512       activation_10[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 16, 107, 128) 0           activation_10[0][0]
                                                                 conv2d_11[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 107, 128) 0           add_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 8, 53, 128)   0           activation_11[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 8, 53, 256)   295168      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 8, 53, 256)   0           conv2d_12[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 8, 53, 256)   590080      activation_12[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 8, 53, 256)   0           conv2d_13[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 8, 53, 256)   65792       activation_13[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 8, 53, 256)   0           activation_13[0][0]
                                                                 conv2d_14[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 8, 53, 256)   0           add_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 4, 26, 256)   0           activation_14[0][0]
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 1, 8, 256)    0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 2048)         0           average_pooling2d[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          524544      flatten[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          32896       dense[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 2)            258         dense_1[0][0]
==================================================================================================
Total params: 1,823,970
Trainable params: 1,823,970
Non-trainable params: 0
__________________________________________________________________________________________________


'''