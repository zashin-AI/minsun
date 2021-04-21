import numpy as np
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
from tensorflow.python.keras.layers.pooling import MaxPooling2D


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
# 리키렐루는 각 뉴런의 출력값이 0보다 높으면 그대로 두고, 0보다 낮으면 정해진 숫자를 곱하는 방식의 함수이다.
# 렐루(ReLU)의 경우 0보다 작을 때 경사가 사라지는 문제가 있지만 리키렐루의 경우 0보다 작을 때도 미분을 적용할 수 있다.
model = Sequential()

activation = 'relu'

inputs = Input(shape=x_train.shape[1:])

# ONE LAYER
def model(output_dim=8, max_length=50, num_filters=5, filter_sizes = [3,5], pooling = 'max', pool_padding = 'valid', dropout = 0.2):
    # Input Layer
#     embed_input = Input(shape=(max_length,output_dim))
    embed_input = Input(shape=(max_length,))
    x = Embedding(vocab_size,output_dim,input_length=max_length)(embed_input)
#     x = SpatialDropout1D(0.2)(x)
    ## concat
    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)
        if pooling=='max':
            conv = MaxPooling1D(pool_size=max_length-filter_sizes[i]+1, strides=1, padding = pool_padding)(conv)
        else:
            conv = AveragePooling1D(pool_size=max_length-filter_sizes[i]+1, strides=1, padding = pool_padding)(conv)            
        pooled_outputs.append(conv)
    merge = concatenate(pooled_outputs)
        
    x = Flatten()(merge)
    x = Dropout(dropout)(x)
#     predictions = Dense(y_dim, activation = 'sigmoid')(x)
    predictions = Dense(2, activation = 'softmax')(x) # TEST
    
    model = Model(inputs=embed_input,outputs=predictions)

    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
    print(model.summary())
    
    from keras.utils import plot_model
    plot_model(model, to_file='shared_input_layer.png')
    
    return model


model = model(output_dim=16, max_length=max_length,y_dim=5,filter_sizes = [3,4,5],pooling = 'max',dropout=0.5)  

    

def residual_block(x, filters, conv_num=4, activation='relu'): 
    for i in range(conv_num - 1):
        x = Conv2D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool2D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    # x = residual_block(x, 128, 3)
    # x = residual_block(x, 128, 3)

    x = AveragePooling2D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)
model = build_model(x_train.shape[1:], 2)
print(x_train.shape[1:])    # (128, 862, 1)
model.summary()

model.save('C:/nmb/nmb_data/h5/Conv2D_model_Adam.h5')

start = datetime.now()
# 컴파일, 훈련
op = Nadam(lr=1e-3)
batch_size = 32

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/Conv2D_weight_NA32_3.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])


# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/Conv2D_weight_NA32_3.h5')
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

pred_pathAudio = 'C:/nmb/nmb_data/predict/F'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
count_f = 0

for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    y_pred = model.predict(pred_mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :
        count_f += 1                 
        print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
    else:                         
        print(file, '{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))

pred_pathAudio = 'C:/nmb/nmb_data/predict/M'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
count_m = 0
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    y_pred = model.predict(pred_mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :               
        print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
    else:
        count_m += 1                              
        print(file, '{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))

print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")


end = datetime.now()
time = end - start
print("작업 시간 : " , time)  


# # import winsound as sd
# # def beepsound():
# #     fr = 440    # range : 37 ~ 32767
# #     du = 500     # 1000 ms ==1second
# #     sd.Beep(fr, du) # winsound.Beep(frequency, duration)

# # beepsound()

'''
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              [(None, 128, 862, 1) 0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 128, 862, 16) 160         input[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 128, 862, 16) 0           conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 128, 862, 16) 2320        activation[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 128, 862, 16) 32          input[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 128, 862, 16) 0           conv2d_2[0][0]
                                                                 conv2d[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 862, 16) 0           add[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 127, 861, 16) 0           activation_1[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 127, 861, 32) 4640        max_pooling2d[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 127, 861, 32) 0           conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 127, 861, 32) 9248        activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 127, 861, 32) 544         max_pooling2d[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 127, 861, 32) 0           conv2d_5[0][0]
                                                                 conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 127, 861, 32) 0           add_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 126, 860, 32) 0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 126, 860, 64) 18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 126, 860, 64) 0           conv2d_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 126, 860, 64) 36928       activation_4[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 126, 860, 64) 0           conv2d_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 126, 860, 64) 36928       activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 126, 860, 64) 2112        max_pooling2d_1[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 126, 860, 64) 0           conv2d_9[0][0]
                                                                 conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 126, 860, 64) 0           add_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 125, 859, 64) 0           activation_6[0][0]
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 41, 286, 64)  0           max_pooling2d_2[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 750464)       0           average_pooling2d[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          192119040   flatten[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          32896       dense[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            258         dense_1[0][0]
==================================================================================================
Total params: 192,263,602
Trainable params: 192,263,602
Non-trainable params: 0
__________________________________________________________________________________________________

'''