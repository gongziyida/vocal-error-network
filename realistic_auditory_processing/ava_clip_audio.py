import os, glob, h5py
import numpy as np
from scipy.io import wavfile
from joblib import Parallel, delayed
from itertools import repeat
from ava.preprocessing.preprocess import get_audio_seg_filenames, read_onsets_offsets_from_file

params = {
    'max_dur': 0.35, # maximum syllable duration
    'len_time_step': 10, # length of a time step in terms of the samples
    'fs': 44100, # audio samplerate
    'max_num_syllables': None, # maximum number of syllables per directory
    'sylls_per_file': 100, # syllable per file
}

def clip_audio_sylls(audio_dir, segment_dir, save_dir, p, shuffle=True, \
    verbose=True):
    """
    Extract syllables from `audio_dir` and save to `save_dir`.

    Parameters
    ----------
    audio_dir : str
        Directory containing audio files.
    segment_dir : str
        Directory containing segmenting decisions.
    save_dir : str
        Directory to save processed syllables in.
    p : dict
        Preprocessing parameters. TO DO: add reference.
    shuffle : bool, optional
        Shuffle by filename. Defaults to ``True``.
    verbose : bool, optional
        Defaults to ``True``.
    """
    if verbose:
        print("Processing audio files in", audio_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    audio_filenames, seg_filenames = \
            get_audio_seg_filenames(audio_dir, segment_dir, p)
    if shuffle:
        np.random.seed(42)
        perm = np.random.permutation(len(audio_filenames))
        np.random.seed(None)
        audio_filenames = np.array(audio_filenames)[perm]
        seg_filenames = np.array(seg_filenames)[perm]
    write_file_num = 0
    # 'specs' is actually audio clips; 
    # for compatibility with get_syllable_data_loaders
    syll_data = {
        'specs':[],
        'onsets':[],
        'offsets':[],
        'audio_filenames':[],
    }
    sylls_per_file = p['sylls_per_file']
    # For each pair of files...
    for audio_filename, seg_filename in zip(audio_filenames, seg_filenames):
        # Get onsets and offsets.
        onsets, offsets = read_onsets_offsets_from_file(seg_filename, p)
        # Retrieve the audio clip for each detected syllable.
        clips = []
        fs, audio = wavfile.read(audio_filename)
        buffer_len = int(np.ceil(p['max_dur'] * fs / p['len_time_step']))
        
        for i, t1, t2 in zip(range(len(onsets)), onsets, offsets):
            s1, s2 = int(round(t1*fs)), int(round(t2*fs))
            buffer = np.zeros(buffer_len)
            aux = audio[max(0,s1):min(len(audio),s2)][::p['len_time_step']]
            buffer[:len(aux)] = aux
            clips.append(buffer.copy())
        
        # Add the syllables to <syll_data>.
        syll_data['specs'] += clips
        syll_data['onsets'] += onsets.tolist()
        syll_data['offsets'] += offsets.tolist()
        syll_data['audio_filenames'] += \
                len(onsets)*[os.path.split(audio_filename)[-1]]
        # Write files until we don't have enough syllables.
        while len(syll_data['onsets']) >= sylls_per_file:
            save_filename = \
                    "syllables_" + str(write_file_num).zfill(4) + '.hdf5'
            save_filename = os.path.join(save_dir, save_filename)
            with h5py.File(save_filename, "w") as f:
                # Add all the fields.
                for key in ['onsets', 'offsets']:
                    f.create_dataset(key, \
                            data=np.array(syll_data[key][:sylls_per_file]))
                f.create_dataset('specs', \
                        data=np.stack(syll_data['specs'][:sylls_per_file]))
                temp = [os.path.join(audio_dir, i) for i in \
                        syll_data['audio_filenames'][:sylls_per_file]]
                f.create_dataset('audio_filenames', \
                        data=np.array(temp).astype('S'))
            write_file_num += 1
            # Remove the written data from temporary storage.
            for key in syll_data:
                syll_data[key] = syll_data[key][sylls_per_file:]
            # Stop if we've written `max_num_syllables`.
            if p['max_num_syllables'] is not None and \
                    write_file_num*sylls_per_file >= p['max_num_syllables']:
                if verbose:
                    print("\tSaved max_num_syllables (" + \
                            str(p['max_num_syllables'])+"). Returning.")
                return
    if verbose:
        print("\tDone.")


# Define directories.
audio_dirs = glob.glob('../audio_data/[0-9]*')
# seg_dirs = glob.glob('../audio_segs/[0-9]*')
seg_dirs = ['../audio_segs/'+_.split('/')[-1] for _ in audio_dirs]
clip_dirs = ['../audio_clips/'+_.split('/')[-1] for _ in audio_dirs]
for clip_dir in clip_dirs:
    if not os.path.exists(clip_dir):
        os.mkdir(clip_dir)

# generate audio clips in .hdf5 format
gen = zip(audio_dirs, seg_dirs, clip_dirs, repeat(params))
Parallel(n_jobs=5)(delayed(clip_audio_sylls)(*args) for args in gen)
