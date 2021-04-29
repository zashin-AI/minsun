import os
import sys
sys.path.append('c:/nmb/nada/python_import/')

from noise_handling import denoise_tim

denoise_tim(
    load_dir = 'C:\\nmb\\nmb_data\\STT\\STT voice\\F_pred\\',
    out_dir = 'C:\\nmb\\nmb_data\\STT\\STT voice\\F_denoise\\',
    noise_min = 5000,
    noise_max = 15000,
    n_fft = 512,
    hop_length = 128,
    win_length = 512
)