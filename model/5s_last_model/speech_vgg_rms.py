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
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop, SGD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from speech_vgg import speechVGG
from keras import backend as K
from tensorflow import keras

# 데이터 불러오기
x = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_data.npy') # data
y = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_label.npy') # label

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=42)

print(x.shape, y.shape) # (4536, 128, 862) (4536,)
print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862) (908,)


x_train = x_train.reshape(3628, 128, 862,1)
x_test = x_test.reshape(908, 128, 862,1)

# 모델 구성
model = speechVGG(
    include_top=True,
    input_shape=(128,862,1),
    classes=2,
    pooling=None,
    weights=None,
    transfer_learning=True
)

model.summary()

# model.save('C:/nmb/nmb_data/h5/5s_last/model_speech_vgg.h5')

start = datetime.now()

op = RMSprop(lr=1e-5)
batch_size = 8

model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/5s_last/speech_vgg_RMS.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ 'speech_vgg_RMS' + "/",histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])


# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/5s_last/speech_vgg_RMS.h5')
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]) + '\n')

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
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0],pred_mels.shape[1])
        # print(pred_mels.shape)

        y_pred = model.predict(pred_mels)
        # print(y_pred)
        y_pred_label = np.argmax(y_pred)
        # print(y_pred_label)

        if y_pred_label == 0:   # 여성이라고 예측
            # print(file[file.rfind('\\') + 1 :], '여자입니다.')
            if name == 'F' :
                count_f += 1
        else:                   # 남성이라고 예측
            # print(file[file.rfind('\\') + 1 :], '남자입니다.')
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
# loss : 0.03696
# acc : 0.98789

# 43개 여성 목소리 중 41개 정답
# 43개 남성 목소리 중 42개 정답
# 작업 시간 :  0:17:02.853631