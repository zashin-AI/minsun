import numpy as np
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,LSTM,Bidirectional, Conv1D,MaxPool1D, AveragePooling1D,Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint
import os
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
from sklearn.preprocessing import MinMaxScaler, StandardScaler



# 데이터 불러오기
x = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_data.npy') # data
y = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_label.npy') # label

x = x.reshape(-1, x.shape[1] * x.shape[2])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=42
)

print(x.shape, y.shape)
# (4536, 128, 862) (4536,)
print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862) (908,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(3628, 128, 862)
x_test = x_test.reshape(908, 128, 862)

# 모델 구성
model = Sequential()

def residual_block(x, units, conv_num=3, activation='tanh'):  
    # Shortcut
    s = LSTM(units, return_sequences=True)(x) 
    for i in range(conv_num - 1):
        x = LSTM(units, return_sequences=True)(x) 
        x = Activation(activation)(x)
    x = LSTM(units)(x)
    x = Add()([x,s])
    return Activation(activation)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = Bidirectional(LSTM(16))(x)  #  LSTM 레이어 부분에 Bidirectional() 함수 -> many to one 유형
    x = Dense(256, activation="tanh")(x)
    x = Dense(128, activation="tanh")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2) # lstm 사용할때는 1로 적용해서 softmax사용후 sparse_categorical_crossentropy 적용 -> 자세한 내용 찾아봐야함
print(x_train.shape[1:])    # (128, 862)

model.summary()

model.save('C:/nmb/nmb_data/h5/5s_last/model_lstm2_adam_mms.h5')

start = datetime.now()

op = Adam(lr=0.001)
batch_size = 32

model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/5s_last/lstm2_adam_mms.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ 'lstm2_adam_mms' + "/",histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])


# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/5s_last/lstm2_adam_mms.h5')
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

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
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0]*pred_mels.shape[1])
        # print(pred_mels.shape)

        pred_mels = scaler.transform(pred_mels) # minmaxscaler
        pred_mels = pred_mels.reshape(1, 128, 862)
        # # print(pred_mels)

        y_pred = model.predict(pred_mels)
        # print(y_pred)
        y_pred_label = np.argmax(y_pred)
        # print(y_pred_label)

        if y_pred_label == 0:   # 여성이라고 예측
            print(file[file.rfind('\\') + 1 :], '여자입니다.')
            if name == 'F' :
                count_f += 1
        else:                   # 남성이라고 예측
            print(file[file.rfind('\\') + 1 :], '남자입니다.')
            if name == 'M' :
                count_m += 1
                
                    
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("43개 남성 목소리 중 "+str(count_m)+"개 정답")


end = datetime.now()
time = end - start
print("작업 시간 : " , time) 

'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              [(None, 128, 862)]   0
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 128, 16)      56256       input[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 128, 16)      0           lstm_1[0][0]
__________________________________________________________________________________________________
lstm_2 (LSTM)                   (None, 16)           2112        activation[0][0]
__________________________________________________________________________________________________
lstm (LSTM)                     (None, 128, 16)      56256       input[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 128, 16)      0           lstm_2[0][0]
                                                                 lstm[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 16)      0           add[0][0]
__________________________________________________________________________________________________
lstm_4 (LSTM)                   (None, 128, 32)      6272        activation_1[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 128, 32)      0           lstm_4[0][0]
__________________________________________________________________________________________________
lstm_5 (LSTM)                   (None, 32)           8320        activation_2[0][0]
__________________________________________________________________________________________________
lstm_3 (LSTM)                   (None, 128, 32)      6272        activation_1[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 128, 32)      0           lstm_5[0][0]
                                                                 lstm_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 128, 32)      0           add_1[0][0]
__________________________________________________________________________________________________
lstm_7 (LSTM)                   (None, 128, 64)      24832       activation_3[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 128, 64)      0           lstm_7[0][0]
__________________________________________________________________________________________________
lstm_8 (LSTM)                   (None, 128, 64)      33024       activation_4[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 128, 64)      0           lstm_8[0][0]
__________________________________________________________________________________________________
lstm_9 (LSTM)                   (None, 64)           33024       activation_5[0][0]
__________________________________________________________________________________________________
lstm_6 (LSTM)                   (None, 128, 64)      24832       activation_3[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 128, 64)      0           lstm_9[0][0]
                                                                 lstm_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 128, 64)      0           add_2[0][0]
__________________________________________________________________________________________________
lstm_11 (LSTM)                  (None, 128, 128)     98816       activation_6[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 128, 128)     0           lstm_11[0][0]
__________________________________________________________________________________________________
lstm_12 (LSTM)                  (None, 128, 128)     131584      activation_7[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 128, 128)     0           lstm_12[0][0]
__________________________________________________________________________________________________
lstm_13 (LSTM)                  (None, 128)          131584      activation_8[0][0]
__________________________________________________________________________________________________
lstm_10 (LSTM)                  (None, 128, 128)     98816       activation_6[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 128, 128)     0           lstm_13[0][0]
                                                                 lstm_10[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 128, 128)     0           add_3[0][0]
__________________________________________________________________________________________________
lstm_15 (LSTM)                  (None, 128, 128)     131584      activation_9[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 128, 128)     0           lstm_15[0][0]
__________________________________________________________________________________________________
lstm_16 (LSTM)                  (None, 128, 128)     131584      activation_10[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 128, 128)     0           lstm_16[0][0]
__________________________________________________________________________________________________
lstm_17 (LSTM)                  (None, 128)          131584      activation_11[0][0]
__________________________________________________________________________________________________
lstm_14 (LSTM)                  (None, 128, 128)     131584      activation_9[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 128, 128)     0           lstm_17[0][0]
                                                                 lstm_14[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 128, 128)     0           add_4[0][0]
__________________________________________________________________________________________________
bidirectional (Bidirectional)   (None, 32)           18560       activation_12[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          8448        bidirectional[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          32896       dense[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            258         dense_1[0][0]
==================================================================================================
Total params: 1,298,498
Trainable params: 1,298,498
Non-trainable params: 0
__________________________________________________________________________________________________
'''