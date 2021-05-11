import os
import sys
sys.path.append('c:/nmb/nada/python_import/')

from noise_handling import denoise_tim_data, denoise_tim_reduce

denoise_tim_data(
    load_dir = 'C:\\nmb\\nmb_data\\STT\\slr_origin\\',
    out_dir = 'C:\\nmb\\nmb_data\\STT\\slr_denoise\\',
    noise_min = 5000,
    noise_max = 15000,
    n_fft = 512,
    hop_length = 128,
    win_length = 512
)

denoise_tim_reduce(
    load_dir = 'C:\\nmb\\nmb_data\\STT\\slr_origin\\',
    out_dir = 'C:\\nmb\\nmb_data\\STT\\slr_reduce\\',
    noise_min = 5000,
    noise_max = 15000,
    n_fft = 512,
    hop_length = 128,
    win_length = 512
)