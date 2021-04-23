
import sys
import os
sys.path.append('c:/nmb/nada/python_import/')
from split_silence import split_silence


audio_dir = 'C:\\nmb\\nmb_data\\audio_sum\\korea_corpus_m_sum\\'
split_silence_dir = "C:\\nmb\\nmb_data\\silence\\split\\korea_corpus_m_silence_split\\"
sum_dir = "C:\\nmb\\nmb_data\\silence\\sum\\korea_corpus_m_silence_sum\\"

split_silence(audio_dir, split_silence_dir, sum_dir)


