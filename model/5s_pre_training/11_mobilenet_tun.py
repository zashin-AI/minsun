

from itertools import count
import numpy as np
import os
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet, InceptionV3, DenseNet121, Xception, EfficientNetB4, VGG16, VGG19, ResNet101, NASNetMobile
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate, LeakyReLU, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
import matplotlib.pyplot as plt

start_now = datetime.now()

# 데이터 불러오기
x = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_data.npy') # data
y = np.load('C:\\nmb\\nmb_data\\npy\\5s_last\\total_label.npy') # label

print(x.shape, y.shape) # (1073, 128, 862) (1073,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

aaa = 1 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3628, 128, 862, 1) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862, 1) (908,)

model = MobileNet(
    include_top=True,
    input_shape=(128,862,1),
    classes=2,
    pooling=None,
    weights='imagenet',
)
# model.trainable = False
model.summary()

model.save('C:\\nmb\\nmb_data\\h5\\pre_train\\11_mobilenet_rmsprop_false_img.h5')

# 컴파일, 훈련
op = RMSprop(lr=1e-3)
batch_size = 8
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:\\nmb\\nmb_data\\h5\\pre_train\\11_mobilenet_rmsprop_false_img.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ '11_mobilenet_rmsprop_false_img' + "/",histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=1000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])

print('\nhistory dict:', history.history)

plt.plot(history.epoch, history.history['loss'], '-o', label='training_loss')
plt.plot(history.epoch, history.history['val_loss'], '-o', label='validation_loss')
plt.legend()
plt.xlim(left=0)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# 평가, 예측
model = load_model('C:\\nmb\\nmb_data\\h5\\pre_train\\11_mobilenet_rmsprop_false_img.h5')
# model.load_weights('C:/nmb/nmb_data/h5/5s/mobilenet/mobilenet_rmsprop_1.h5')
result = model.evaluate(x_test, y_test, batch_size=8)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))


############################################ PREDICT ####################################

pred = ['C:/nmb/nmb_data/predict/5s_last/F','C:/nmb/nmb_data/predict/5s_last/M']

count_f = 0
count_m = 0

for pred_pathAudio in pred:
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
time = end - start_now
print("작업 시간 : ", time)

# trainable = true , weight = none
# loss : 0.11121
# acc : 0.97797
# 43개 여성 목소리 중 41개 정답
# 43개 남성 목소리 중 42개 정답
# 작업 시간 :  0:25:27.264733

#------------------------------------------------------------------------------------

# trainable = False, weight = none
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 128, 862, 1)]     0
_________________________________________________________________
conv1 (Conv2D)               (None, 64, 431, 32)       288
_________________________________________________________________
conv1_bn (BatchNormalization (None, 64, 431, 32)       128
_________________________________________________________________
conv1_relu (ReLU)            (None, 64, 431, 32)       0
_________________________________________________________________
conv_dw_1 (DepthwiseConv2D)  (None, 64, 431, 32)       288
_________________________________________________________________
conv_dw_1_bn (BatchNormaliza (None, 64, 431, 32)       128
_________________________________________________________________
conv_dw_1_relu (ReLU)        (None, 64, 431, 32)       0
_________________________________________________________________
conv_pw_1 (Conv2D)           (None, 64, 431, 64)       2048
_________________________________________________________________
conv_pw_1_bn (BatchNormaliza (None, 64, 431, 64)       256
_________________________________________________________________
conv_pw_1_relu (ReLU)        (None, 64, 431, 64)       0
_________________________________________________________________
conv_pad_2 (ZeroPadding2D)   (None, 65, 432, 64)       0
_________________________________________________________________
conv_dw_2 (DepthwiseConv2D)  (None, 32, 215, 64)       576
_________________________________________________________________
conv_dw_2_bn (BatchNormaliza (None, 32, 215, 64)       256
_________________________________________________________________
conv_dw_2_relu (ReLU)        (None, 32, 215, 64)       0
_________________________________________________________________
conv_pw_2 (Conv2D)           (None, 32, 215, 128)      8192
_________________________________________________________________
conv_pw_2_bn (BatchNormaliza (None, 32, 215, 128)      512
_________________________________________________________________
conv_pw_2_relu (ReLU)        (None, 32, 215, 128)      0
_________________________________________________________________
conv_dw_3 (DepthwiseConv2D)  (None, 32, 215, 128)      1152
_________________________________________________________________
conv_dw_3_bn (BatchNormaliza (None, 32, 215, 128)      512
_________________________________________________________________
conv_dw_3_relu (ReLU)        (None, 32, 215, 128)      0
_________________________________________________________________
conv_pw_3 (Conv2D)           (None, 32, 215, 128)      16384
_________________________________________________________________
conv_pw_3_bn (BatchNormaliza (None, 32, 215, 128)      512
_________________________________________________________________
conv_pw_3_relu (ReLU)        (None, 32, 215, 128)      0
_________________________________________________________________
conv_pad_4 (ZeroPadding2D)   (None, 33, 216, 128)      0
_________________________________________________________________
conv_dw_4 (DepthwiseConv2D)  (None, 16, 107, 128)      1152
_________________________________________________________________
conv_dw_4_bn (BatchNormaliza (None, 16, 107, 128)      512
_________________________________________________________________
conv_dw_4_relu (ReLU)        (None, 16, 107, 128)      0
_________________________________________________________________
conv_pw_4 (Conv2D)           (None, 16, 107, 256)      32768
_________________________________________________________________
conv_pw_4_bn (BatchNormaliza (None, 16, 107, 256)      1024
_________________________________________________________________
conv_pw_4_relu (ReLU)        (None, 16, 107, 256)      0
_________________________________________________________________
conv_dw_5 (DepthwiseConv2D)  (None, 16, 107, 256)      2304
_________________________________________________________________
conv_dw_5_bn (BatchNormaliza (None, 16, 107, 256)      1024
_________________________________________________________________
conv_dw_5_relu (ReLU)        (None, 16, 107, 256)      0
_________________________________________________________________
conv_pw_5 (Conv2D)           (None, 16, 107, 256)      65536
_________________________________________________________________
conv_pw_5_bn (BatchNormaliza (None, 16, 107, 256)      1024
_________________________________________________________________
conv_pw_5_relu (ReLU)        (None, 16, 107, 256)      0
_________________________________________________________________
conv_pad_6 (ZeroPadding2D)   (None, 17, 108, 256)      0
_________________________________________________________________
conv_dw_6 (DepthwiseConv2D)  (None, 8, 53, 256)        2304
_________________________________________________________________
conv_dw_6_bn (BatchNormaliza (None, 8, 53, 256)        1024
_________________________________________________________________
conv_dw_6_relu (ReLU)        (None, 8, 53, 256)        0
_________________________________________________________________
conv_pw_6 (Conv2D)           (None, 8, 53, 512)        131072
_________________________________________________________________
conv_pw_6_bn (BatchNormaliza (None, 8, 53, 512)        2048
_________________________________________________________________
conv_pw_6_relu (ReLU)        (None, 8, 53, 512)        0
_________________________________________________________________
conv_dw_7 (DepthwiseConv2D)  (None, 8, 53, 512)        4608
_________________________________________________________________
conv_dw_7_bn (BatchNormaliza (None, 8, 53, 512)        2048
_________________________________________________________________
conv_dw_7_relu (ReLU)        (None, 8, 53, 512)        0
_________________________________________________________________
conv_pw_7 (Conv2D)           (None, 8, 53, 512)        262144
_________________________________________________________________
conv_pw_7_bn (BatchNormaliza (None, 8, 53, 512)        2048
_________________________________________________________________
conv_pw_7_relu (ReLU)        (None, 8, 53, 512)        0
_________________________________________________________________
conv_dw_8 (DepthwiseConv2D)  (None, 8, 53, 512)        4608
_________________________________________________________________
conv_dw_8_bn (BatchNormaliza (None, 8, 53, 512)        2048
_________________________________________________________________
conv_dw_8_relu (ReLU)        (None, 8, 53, 512)        0
_________________________________________________________________
conv_pw_8 (Conv2D)           (None, 8, 53, 512)        262144
_________________________________________________________________
conv_pw_8_bn (BatchNormaliza (None, 8, 53, 512)        2048
_________________________________________________________________
conv_pw_8_relu (ReLU)        (None, 8, 53, 512)        0
_________________________________________________________________
conv_dw_9 (DepthwiseConv2D)  (None, 8, 53, 512)        4608
_________________________________________________________________
conv_dw_9_bn (BatchNormaliza (None, 8, 53, 512)        2048
_________________________________________________________________
conv_dw_9_relu (ReLU)        (None, 8, 53, 512)        0
_________________________________________________________________
conv_pw_9 (Conv2D)           (None, 8, 53, 512)        262144
_________________________________________________________________
conv_pw_9_bn (BatchNormaliza (None, 8, 53, 512)        2048
_________________________________________________________________
conv_pw_9_relu (ReLU)        (None, 8, 53, 512)        0
_________________________________________________________________
conv_dw_10 (DepthwiseConv2D) (None, 8, 53, 512)        4608
_________________________________________________________________
conv_dw_10_bn (BatchNormaliz (None, 8, 53, 512)        2048
_________________________________________________________________
conv_dw_10_relu (ReLU)       (None, 8, 53, 512)        0
_________________________________________________________________
conv_pw_10 (Conv2D)          (None, 8, 53, 512)        262144
_________________________________________________________________
conv_pw_10_bn (BatchNormaliz (None, 8, 53, 512)        2048
_________________________________________________________________
conv_pw_10_relu (ReLU)       (None, 8, 53, 512)        0
_________________________________________________________________
conv_dw_11 (DepthwiseConv2D) (None, 8, 53, 512)        4608
_________________________________________________________________
conv_dw_11_bn (BatchNormaliz (None, 8, 53, 512)        2048
_________________________________________________________________
conv_dw_11_relu (ReLU)       (None, 8, 53, 512)        0
_________________________________________________________________
conv_pw_11 (Conv2D)          (None, 8, 53, 512)        262144
_________________________________________________________________
conv_pw_11_bn (BatchNormaliz (None, 8, 53, 512)        2048
_________________________________________________________________
conv_pw_11_relu (ReLU)       (None, 8, 53, 512)        0
_________________________________________________________________
conv_pad_12 (ZeroPadding2D)  (None, 9, 54, 512)        0
_________________________________________________________________
conv_dw_12 (DepthwiseConv2D) (None, 4, 26, 512)        4608      
_________________________________________________________________
conv_dw_12_bn (BatchNormaliz (None, 4, 26, 512)        2048
_________________________________________________________________
conv_dw_12_relu (ReLU)       (None, 4, 26, 512)        0
_________________________________________________________________
conv_pw_12 (Conv2D)          (None, 4, 26, 1024)       524288
_________________________________________________________________
conv_pw_12_bn (BatchNormaliz (None, 4, 26, 1024)       4096
_________________________________________________________________
conv_pw_12_relu (ReLU)       (None, 4, 26, 1024)       0
_________________________________________________________________
conv_dw_13 (DepthwiseConv2D) (None, 4, 26, 1024)       9216
_________________________________________________________________
conv_dw_13_bn (BatchNormaliz (None, 4, 26, 1024)       4096
_________________________________________________________________
conv_dw_13_relu (ReLU)       (None, 4, 26, 1024)       0
_________________________________________________________________
conv_pw_13 (Conv2D)          (None, 4, 26, 1024)       1048576
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 4, 26, 1024)       4096
_________________________________________________________________
conv_pw_13_relu (ReLU)       (None, 4, 26, 1024)       0
_________________________________________________________________
global_average_pooling2d (Gl (None, 1024)              0
_________________________________________________________________
reshape_1 (Reshape)          (None, 1, 1, 1024)        0
_________________________________________________________________
dropout (Dropout)            (None, 1, 1, 1024)        0
_________________________________________________________________
conv_preds (Conv2D)          (None, 1, 1, 2)           2050
_________________________________________________________________
reshape_2 (Reshape)          (None, 2)                 0
_________________________________________________________________
predictions (Activation)     (None, 2)                 0
=================================================================
Total params: 3,230,338
Trainable params: 0
Non-trainable params: 3,230,338
_________________________________________________________________
'''
# loss : 0.69315
# acc : 0.55066
# 43개 여성 목소리 중 43개 정답
# 43개 남성 목소리 중 0개 정답
# 작업 시간 :  0:01:35.931280

# weight='imagenet'
# If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000

# include_top = False
# ValueError: The input must have 3 channels; got `input_shape=(128, 862, 1)`