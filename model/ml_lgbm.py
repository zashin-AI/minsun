
# LGBM 설치
# conda install - c conda-forge lightgbm
# pip install lightgbm

from lightgbm import LGBMClassifier, plot_importance
import numpy as np
import datetime 
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error
# from sklearn.utils import all_estimators  
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/brandnew_0_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/brandnew_0_mels_label.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/brandnew_1_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/brandnew_1_mels_label.npy')

x = np.concatenate([f_ds, m_ds], 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)  # (2141, 110336)
print(y.shape)  # (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (1712, 110336)
print(x_test.shape)     # (429, 110336)
print(y_train.shape)    # (1712,)
print(y_test.shape)     # (429,)

# 모델 구성
# model = LGBMClassifier(n_estimators= 10000)
# evals = [(x_test, y_test)]
# model.fit(x_train, y_train, early_stopping_rounds= 100, eval_metric= 'logloss', eval_set=evals, verbose=True)
# model.fit(x_train, y_train,  eval_metric= 'logloss')
model = LGBMClassifier()
model.fit(x_train, y_train)

# model & weight save
pickle.dump(model, open('C:/nmb/nmb_data/h5/LGBM0.data', 'wb')) # wb : write
# print("== save complete ==")

# model load
# model = pickle.load(open('E:/nmb/nmb_data/cp/m03_mels_SVC.data', 'rb'))  # rb : read
# time >>  0:01:07.868304

# evaluate
y_pred = model.predict(x_test)
# print(y_pred[:100])
# print(y_pred[100:])

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
hamm_loss = hamming_loss(y_test, y_pred)
hinge_loss = hinge_loss(y_test, y_pred)
log_loss = log_loss(y_test, y_pred)

print("accuracy : \t", accuracy)
print("recall : \t", recall)
print("precision : \t", precision)
print("f1 : \t", f1)

print("hamming_loss : \t", hamm_loss)
print("hinge_loss : \t", hinge_loss)                    # SVM에 적합한 cross-entropy
print("log_loss : \t", log_loss)                        # Cross-entropy loss와 유사한 개념
print("mse : \t", mean_squared_error(y_test, y_pred))   # Regression 모델에서의 loss



# predict 데이터
pred_pathAudio = 'C:/nmb/nmb_data/pred_voice/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    pred_mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
    pred_mels = librosa.amplitude_to_db(pred_mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0] * pred_mels.shape[1])
    # print(pred_mels.shape)  # (1, 110336)
    y_pred = model.predict(pred_mels)
    # print(y_pred)
    if y_pred == 0 :                    # label 0
        print(file, '여자입니다.')
    else:                               # label 1
        print(file, '남자입니다.')


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >


# model = LGBMClassifier(n_estimators=400)
'''
accuracy :       0.9370629370629371
recall :         0.9658536585365853
precision :      0.908256880733945
f1 :     0.9361702127659574
hamming_loss :   0.06293706293706294
hinge_loss :     0.585081585081585
log_loss :       2.173806421005111
mse :    0.06293706293706294
C:\nmb\nmb_data\pred_voice\FY1.wav 여자입니다.
C:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 남자입니다.
time >>  0:04:20.634776
'''
# model = LGBMClassifier(n_estimators=4000)
# 기본
'''
accuracy :       0.9487179487179487
recall :         0.9853658536585366
precision :      0.9140271493212669
f1 :     0.9483568075117371
hamming_loss :   0.05128205128205128
hinge_loss :     0.5734265734265734
log_loss :       1.771254715709112
mse :    0.05128205128205128
C:\nmb\nmb_data\pred_voice\FY1.wav 여자입니다.
C:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 여자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 남자입니다.
time >>  0:03:05.256056
'''