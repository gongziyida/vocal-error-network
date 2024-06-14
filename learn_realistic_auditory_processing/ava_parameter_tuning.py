import pickle
from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from ava.segmenting.segment import tune_segmenting_params

raw_seg_params = {
    'min_freq': 5e2, # minimum frequency
    'max_freq': 10e3, # maximum frequency
    'nperseg': 512, # FFT
    'noverlap': 256, # FFT
    'spec_min_val': 1.0, # minimum STFT log-modulus
    'spec_max_val': 6.0, # maximum STFT log-modulus
    'fs': 44100, # audio samplerate
    'th_1':5, # segmenting threshold 1
    'th_2':10, # segmenting threshold 2
    'th_3': 18, # segmenting threshold 3
    'min_dur':0.05, # minimum syllable duration
    'max_dur': 0.35, # maximum syllable duration
    'smoothing_timescale': 0.001, # amplitude
    'softmax': False, # apply softmax to the frequency bins to calculate
                      # amplitude
    'temperature':0.5, # softmax temperature parameter
    'algorithm': get_onsets_offsets, # (defined above)
}

early_audio_directories = ['../audio_data/40']
late_audio_directories = ['../audio_data/140']

early_seg_params = tune_segmenting_params(early_audio_directories, raw_seg_params.copy())
late_seg_params = tune_segmenting_params(late_audio_directories, raw_seg_params.copy())

with open('../audio_data/seg_params', 'wb') as f:
    pickle.dump([early_seg_params, late_seg_params], f)