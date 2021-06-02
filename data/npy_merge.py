import numpy as np

dir = 'C:/nmb/nmb_data/0418_balance_denoise_npy/'
npy = '.npy'

# 데이터 불러오기 female
f1 = np.load(dir + 'korea_corpus_f_2m_denoise_mels' + npy)
f2 = np.load(dir + 'mindslab_f_2m_denoise_mels' + npy)
f3 = np.load(dir + 'open_slr_f_2m_denoise_mels' + npy)
f4 = np.load(dir + 'pansori_f_2m_denoise_mels' + npy)

# 데이터 불러오기 male
m1 = np.load(dir + 'korea_corpus_m_2m_denoise_mels' + npy)
m2 = np.load(dir + 'mindslab_m_2m_denoise_mels' + npy)
m3 = np.load(dir + 'open_slr_m_2m_denoise_mels' + npy)
m4 = np.load(dir + 'pansori_m_2m_denoise_mels' + npy)

# 라벨 불러오기 female
fl1 = np.load(dir + 'korea_corpus_f_2m_denoise_label_mels' + npy)
fl2 = np.load(dir + 'mindslab_f_2m_denoise_label_mels' + npy)
fl3 = np.load(dir + 'open_slr_f_2m_denoise_label_mels' + npy)
fl4 = np.load(dir + 'pansori_f_2m_denoise_label_mels' + npy)

# 라벨 불러오기 male
ml1 = np.load(dir + 'korea_corpus_m_2m_denoise_label_mels' + npy)
ml2 = np.load(dir + 'mindslab_m_2m_denoise_label_mels' + npy)
ml3 = np.load(dir + 'open_slr_m_2m_denoise_label_mels' + npy)
ml4 = np.load(dir + 'pansori_m_2m_denoise_label_mels' + npy)

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
np.save('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_f_mels.npy', arr=female)
np.save('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_f_label_mels.npy', arr=f_label)
print('=====save done(f)=====')

# 내보내기 male
np.save('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_m_mels.npy', arr=male)
np.save('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_m_label_mels.npy', arr=m_label)
print('=====save done(m)=====')