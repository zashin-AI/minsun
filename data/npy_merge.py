import numpy as np

dir = 'D:/nmb/0602_10s/'
npy = '.npy'

# 데이터 불러오기 female
f1 = np.load(dir + 'koreacorpus_2m10s_f_mel_data' + npy)
f2 = np.load(dir + 'mindslab_2m10s_f_mel_data' + npy)
f3 = np.load(dir + 'open_slr_f_data' + npy)
f4 = np.load(dir + 'pansori_2m10s_f_mel_data' + npy)

# 데이터 불러오기 male
m1 = np.load(dir + 'koreacorpus_2m10s_m_mel_data' + npy)
m2 = np.load(dir + 'mindslab_2m10s_m_mel_data' + npy)
m3 = np.load(dir + 'open_slr_m_data' + npy)
m4 = np.load(dir + 'pansori_2m10s_m_mel_data' + npy)

# 라벨 불러오기 female
fl1 = np.load(dir + 'koreacorpus_2m10s_f_mel_label' + npy)
fl2 = np.load(dir + 'mindslab_2m10s_f_mel_label' + npy)
fl3 = np.load(dir + 'open_slr_f_label' + npy)
fl4 = np.load(dir + 'pansori_2m10s_f_mel_label' + npy)

# 라벨 불러오기 male
ml1 = np.load(dir + 'koreacorpus_2m10s_m_mel_label' + npy)
ml2 = np.load(dir + 'mindslab_2m10s_m_mel_label' + npy)
ml3 = np.load(dir + 'open_slr_m_label' + npy)
ml4 = np.load(dir + 'pansori_2m10s_m_mel_label' + npy)

# 합치기 female
female = np.concatenate([f1, f2, f3, f4], 0)
f_label = np.concatenate([fl1, fl2, fl3, fl4], 0)
print(female.shape)
print(f_label.shape)
print(np.unique(f_label))

# 합치기 male
male = np.concatenate([m1, m2, m3, m4], 0)
m_label = np.concatenate([ml1, ml2, ml3, ml4], 0)
print(male.shape)
print(m_label.shape)
print(np.unique(m_label))


# 내보내기 female
np.save('D:/nmb/0602_10s/total_10s/10s_f_mels.npy', arr=female)
np.save('D:/nmb/0602_10s/total_10s/10s_f_mels_label.npy', arr=f_label)
print('=====save done(f)=====')

# 내보내기 male
np.save('D:/nmb/0602_10s/total_10s/10s_m_mels.npy', arr=male)
np.save('D:/nmb/0602_10s/total_10s/10s_m_mels_label.npy', arr=m_label)
print('=====save done(m)=====')