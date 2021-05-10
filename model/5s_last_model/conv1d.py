import numpy as np
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D,MaxPool1D, AveragePooling1D,Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint
import os
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop



# 데이터 불러오기
x = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_data.npy') # data
y = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_label.npy') # label


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=42
)

print(x.shape, y.shape)
# (4536, 128, 862) (4536,)
print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862) (908,)

# 모델 구성
model = Sequential()

def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding="same")(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=1)(x)


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()

start = datetime.now()

op = Adam(lr=0.001)
batch_size = 32

model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/5s_last/model_Conv1D_adam.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])


# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/5s_last/model_Conv1D_adam.h5')
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : ", result[0])
print("acc : ", result[1])
############################################ PREDICT ####################################
pred = ['C:/nmb/nmb_data/predict/5s_last/F','C:/nmb/nmb_data/predict/5s_last/M']

count_f = 0
count_m = 0

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
            count_f = count_f + 1
                
        else:                   # 남성이라고 예측              
            print(file,'{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))
            count_m = count_m + 1
                
                    
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("43개 남성 목소리 중 "+str(count_m)+"개 정답")


end = datetime.now()
time = end - start
print("time >> " , time)    # time >>  0:00:33.975135

'''
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              [(None, 128, 862)]   0
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 128, 16)      41392       input[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 128, 16)      0           conv1d_1[0][0]
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 128, 16)      784         activation[0][0]
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 128, 16)      13808       input[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 128, 16)      0           conv1d_2[0][0]
                                                                 conv1d[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 16)      0           add[0][0]
__________________________________________________________________________________________________
max_pooling1d (MaxPooling1D)    (None, 127, 16)      0           activation_1[0][0]
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 127, 32)      1568        max_pooling1d[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 127, 32)      0           conv1d_4[0][0]
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 127, 32)      3104        activation_2[0][0]
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 127, 32)      544         max_pooling1d[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 127, 32)      0           conv1d_5[0][0]
                                                                 conv1d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 127, 32)      0           add_1[0][0]
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 126, 32)      0           activation_3[0][0]
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 126, 64)      6208        max_pooling1d_1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 126, 64)      0           conv1d_7[0][0]
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 126, 64)      12352       activation_4[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 126, 64)      0           conv1d_8[0][0]
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 126, 64)      12352       activation_5[0][0]
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 126, 64)      2112        max_pooling1d_1[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 126, 64)      0           conv1d_9[0][0]
                                                                 conv1d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 126, 64)      0           add_2[0][0]
__________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)  (None, 125, 64)      0           activation_6[0][0]
__________________________________________________________________________________________________
conv1d_11 (Conv1D)              (None, 125, 128)     24704       max_pooling1d_2[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 125, 128)     0           conv1d_11[0][0]
__________________________________________________________________________________________________
conv1d_12 (Conv1D)              (None, 125, 128)     49280       activation_7[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 125, 128)     0           conv1d_12[0][0]
__________________________________________________________________________________________________
conv1d_13 (Conv1D)              (None, 125, 128)     49280       activation_8[0][0]
__________________________________________________________________________________________________
conv1d_10 (Conv1D)              (None, 125, 128)     8320        max_pooling1d_2[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 125, 128)     0           conv1d_13[0][0]
                                                                 conv1d_10[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 125, 128)     0           add_3[0][0]
__________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)  (None, 124, 128)     0           activation_9[0][0]
__________________________________________________________________________________________________
conv1d_15 (Conv1D)              (None, 124, 128)     49280       max_pooling1d_3[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 124, 128)     0           conv1d_15[0][0]
__________________________________________________________________________________________________
conv1d_16 (Conv1D)              (None, 124, 128)     49280       activation_10[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 124, 128)     0           conv1d_16[0][0]
__________________________________________________________________________________________________
conv1d_17 (Conv1D)              (None, 124, 128)     49280       activation_11[0][0]
__________________________________________________________________________________________________
conv1d_14 (Conv1D)              (None, 124, 128)     16512       max_pooling1d_3[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 124, 128)     0           conv1d_17[0][0]
                                                                 conv1d_14[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 124, 128)     0           add_4[0][0]
__________________________________________________________________________________________________
max_pooling1d_4 (MaxPooling1D)  (None, 123, 128)     0           activation_12[0][0]
__________________________________________________________________________________________________
average_pooling1d (AveragePooli (None, 41, 128)      0           max_pooling1d_4[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 5248)         0           average_pooling1d[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          1343744     flatten[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          32896       dense[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            258         dense_1[0][0]
==================================================================================================
Total params: 1,767,058
Trainable params: 1,767,058
Non-trainable params: 0
__________________________________________________________________________________________________
'''