import os, glob
from joblib import Parallel, delayed
from itertools import repeat
from ava.preprocessing.utils import get_spec # makes spectrograms
from ava.preprocessing.preprocess import tune_syll_preprocessing_params
from ava.preprocessing.preprocess import process_sylls

preprocess_params = {
    'get_spec': get_spec, # spectrogram maker
    'max_dur': 0.35, # maximum syllable duration
    'min_freq': 5e2, # minimum frequency
    'max_freq': 10e3, # maximum frequency
    'num_freq_bins': 50, # downsample using interpolation
    'num_time_bins': 20,
    'nperseg': 256, # FFT
    'noverlap': 128, # FFT
    'spec_min_val': 2.0, # minimum log-spectrogram value
    'spec_max_val': 6.5, # maximum log-spectrogram value
    'fs': 44100, # audio samplerate
    'mel': False, # frequency spacing, mel or linear
    'time_stretch': True, # stretch short syllables?
    'within_syll_normalize': False, # normalize spectrogram values on a
                                    # spectrogram-by-spectrogram basis
    'max_num_syllables': None, # maximum number of syllables per directory
    'sylls_per_file': 100, # syllable per file
    'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', 
                               'spec_max_val', 'max_dur'), # tunable parameters
    'int_preprocess_params': ('nperseg','noverlap', 'num_freq_bins', 
                              'num_time_bins'), # tunable parameters
    'binary_preprocess_params': ('time_stretch', 'mel', 
                                 'within_syll_normalize'), # tunable parameters
}

# Define directories.
audio_dirs = glob.glob('../audio_data/[0-9]*')
# seg_dirs = glob.glob('../audio_segs/[0-9]*')
seg_dirs = ['../audio_segs/'+_.split('/')[-1] for _ in audio_dirs]
spec_dirs = ['../audio_spec/'+_.split('/')[-1] for _ in audio_dirs]
for spec_dir in spec_dirs:
    if not os.path.exists(spec_dir):
        os.mkdir(spec_dir)

preprocess_params = tune_syll_preprocessing_params(audio_dirs, seg_dirs, \
                preprocess_params)

# generate spectrogram images in .hdf5 format
gen = zip(audio_dirs, seg_dirs, spec_dirs, repeat(preprocess_params))
Parallel(n_jobs=5)(delayed(process_sylls)(*args) for args in gen)