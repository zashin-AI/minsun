
import librosa
from pydub import AudioSegment
import soundfile as sf
import os

# example
# form: 'wav' or 'flac'
# pathaudio = 'C:/nmb/nmb_data/F1F2F3/F3/'
# save_dir = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'

def audiosplit(form, pathaudio, save_dir, out_dir):
    if form =='flac':
        infiles = librosa.util.find_files(pathaudio)
        for infile in infiles:
            # flac 파일의 이름 불러오기
            _, w_id = os.path.split(infile)
            # 이름에서 뒤에 .flac 떼기
            w_id = w_id[:-5]
            # flac 파일의 data, sr 불러오기
            w_data, w_sr = sf.read(infile)
            # 같은 이름으로 wav 형식으로 저장하기
            sf.write(save_dir + w_id + '.wav', w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
        print('==== flac to wav done ====')
        infiles = librosa.util.find_files(save_dir)
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        combined = wavs[0]
        for wav in wavs[1:]:
            combined = combined.append(wav) 
        combined.export(out_dir, format='wav')
        print('==== wav save done ====')

    if form == 'wav':
        infiles = librosa.util.find_files(pathaudio)
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        combined = wavs[0]
        for wav in wavs[1:]:
            combined = combined.append(wav) 
        combined.export(out_dir, format='wav')
    print('==== wav save done ====')