import glob
from ava.segmenting.template_segmentation import get_template
from ava.segmenting.template_segmentation import read_segment_decisions
from ava.segmenting.template_segmentation import clean_collected_segments

params = {
        'min_freq': 1e3, # minimum frequency
        'max_freq': 10e3, # maximum frequency
        'nperseg': 1024, # FFT
        'noverlap': 512, # FFT
        'spec_min_val': 2.5, # minimum spectrogram value
        'spec_max_val': 6.0, # maximum spectrogram value
        'fs': 44100, # audio samplerate
}

template_dir = '../template_songs'
template = get_template(template_dir, params)

audio_dirs = glob.glob('../audio_data/1[0-9]*')
seg_dirs = ['../audio_song_segs/'+_.split('/')[-1] for _ in audio_dirs]

result = read_segment_decisions(audio_dirs, seg_dirs)
clean_collected_segments(result, audio_dirs, seg_dirs, params)