import numpy as np
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,LSTM, Conv1D,MaxPool1D, AveragePooling1D,Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
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
model.add(LSTM(256, return_sequences=True, input_shape= (128,862)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.save('C:/nmb/nmb_data/h5/5s_last/model_lstm_adam_mms.h5')

start = datetime.now()

op = Adam(lr=0.001)
batch_size = 32

model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/5s_last/lstm_adam_mms.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ 'lstm_adam_mms' + "/",histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])


# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/5s_last/lstm_adam_mms.h5')
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
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 128, 256)          1145856
_________________________________________________________________
lstm_1 (LSTM)                (None, 128, 128)          197120
_________________________________________________________________
lstm_2 (LSTM)                (None, 64)                49408
_________________________________________________________________
dense (Dense)                (None, 64)                4160
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 66
=================================================================
Total params: 1,398,690
Trainable params: 1,398,690
Non-trainable params: 0
_________________________________________________________________
'''