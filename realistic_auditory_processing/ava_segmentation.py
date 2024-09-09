import os, glob, pickle, shutil
from joblib import Parallel, delayed
from itertools import repeat
from ava.segmenting.segment import segment

audio_dirs = glob.glob('../audio_data/[0-9]*')
seg_dirs = ['../audio_segs/'+_.split('/')[-1] for _ in audio_dirs]
for seg_dir in seg_dirs:
    if os.path.exists(seg_dir):
        shutil.rmtree(seg_dir)
        os.mkdir(seg_dir)

audio_dirs1 = [_ for _ in audio_dirs if int(_.split('/')[-1])<100] # young
audio_dirs2 = [_ for _ in audio_dirs if int(_.split('/')[-1])>100] # mature
seg_dirs1 = [_ for _ in seg_dirs if int(_.split('/')[-1])<100]
seg_dirs2 = [_ for _ in seg_dirs if int(_.split('/')[-1])>100]
        
with open('../audio_data/seg_params', 'rb') as f:
    early_seg_params, late_seg_params = pickle.load(f)

## Sequential
# for audio_dir, seg_dir in zip(audio_dirs, seg_dirs1):
#     segment(audio_dir, seg_dir, early_seg_params)
# for audio_dir, seg_dir in zip(audio_dirs, seg_dirs2):
#     segment(audio_dir, seg_dir, late_seg_params)

gen1 = zip(audio_dirs1, seg_dirs1, repeat(early_seg_params))
Parallel(n_jobs=5)(delayed(segment)(*args1) for args1 in gen1)

gen2 = zip(audio_dirs2, seg_dirs2, repeat(late_seg_params))
Parallel(n_jobs=2)(delayed(segment)(*args2) for args2 in gen2)