
import sys
import os
sys.path.append('c:/nmb/nada/python_import/')
from split_silence import split_silence


audio_dir = 'C:\\nmb\\nmb_data\\audio_sum\\open_slr_f_sum\\'
split_silence_dir = "C:\\nmb\\nmb_data\\silence\\split\\open_slr_f_silence_split2\\"
sum_dir = "C:\\nmb\\nmb_data\\silence\\sum\\open_slr_f_silence_sum2\\"

split_silence(audio_dir, split_silence_dir, sum_dir)


