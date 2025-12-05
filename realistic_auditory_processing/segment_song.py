import numpy as np
from scipy.io import loadmat, wavfile
from scipy.interpolate import RegularGridInterpolator
import re, glob
import numpy as np
from scipy.signal import stft
from scipy.io import loadmat, wavfile

rng = np.random.default_rng()
NOTE_FILES = glob.glob('../adult_songs_raw/*/R*.not.mat')
CODE = 'iabcde'
pert_syl_idx = 2
print('Found %d files.' % len(NOTE_FILES))

def get_spec(audio, fs, max_dur, min_freq, max_freq, num_freq_bins, num_time_bins, 
             spec_min, spec_max, nperseg=1024, noverlap=512, l_pad=None, r_pad=None):
    T = len(audio) / fs # total time
    audio = audio - np.mean(audio) # remove dc component
    f, t, spec = stft(audio, fs=fs, nperseg=nperseg, noverlap=noverlap)
    spec = np.log(np.abs(spec) + 1e-12)

    # padding
    if max_dur < T:
        print('Audio length %.1f ms > the max duration %.1f ms' % (T*1e3, max_dur*1e3))
        l_pad, r_pad, T = 0, 0, max_dur
    _pad = (max_dur - T) / 2
    l_pad = _pad if l_pad is None else l_pad
    r_pad = _pad if r_pad is None else r_pad
    
    # interpolation
    interp = RegularGridInterpolator((f, t), spec, bounds_error=False, fill_value=-1e12)
    target_freqs = np.linspace(min_freq, max_freq, num_freq_bins)
    target_times = np.linspace(0-l_pad, T+r_pad, num_time_bins)
    tnew, fnew = np.meshgrid(target_times, target_freqs, indexing='xy', sparse=True)
    spec = interp((fnew, tnew))
    
    # normalize
    spec = (spec - spec_min) / (spec_max - spec_min)
    spec = np.clip(spec, 0.0, 1.0)
    return spec
    
spec_song, spec_song_weak_pert, spec_song_strong_pert = [], [], []
spec_syl, spec_syl_weak_pert, spec_syl_strong_pert = [], [], []
syl_on, syl_off, song_Ts = [], [], []
song_kwargs = dict(min_freq=1e3, max_freq=8e3, num_freq_bins=50, spec_min=3.5, spec_max=8, 
                   num_time_bins=80, max_dur=1.1, l_pad=0, r_pad=0)
syl_kwargs = dict(min_freq=1e3, max_freq=8e3, num_freq_bins=50, spec_min=3.5, spec_max=8, 
                  num_time_bins=20, max_dur=0.2)
for f in NOTE_FILES:
    notes = loadmat(f)
    print(notes['labels'])
    start_idx = np.array([m.start() for m in re.finditer(CODE, str(notes['labels'][0]))])
    if len(start_idx) == 0:
        continue
    fs, audio = wavfile.read(f.replace('.not.mat', ''))
    for i in start_idx:
        # onset and offset, in ms
        t0 = notes['onsets'][i:i+len(CODE),0]
        t1 = notes['offsets'][i:i+len(CODE),0]
        T = notes['offsets'][i+len(CODE)-1,0] - notes['onsets'][i,0]
        song_Ts.append(T)
        syl_on.append(t0 - t0[0])
        syl_off.append(t1 - t0[0])

        # onset and offset global index
        t0_, t1_ = (t0/1e3*fs).astype(int), (t1/1e3*fs).astype(int)
        
        # song spectrum
        song = audio[t0_[0]:t1_[-1]].astype(np.float64)
        if len(song)/fs > 1.1:
            print(len(spec_song))
        
        spec_song.append(get_spec(song, fs, **song_kwargs))

        # song spectrum with perturbation between pert_t0 and pert_t1
        ts = np.linspace(0, T, endpoint=False, num=len(song)) # ms
        pert_t0 = t0[pert_syl_idx] - t0[0]
        pert_t1 = min(pert_t0+50, t1[pert_syl_idx]-t0[0])
        mask = (ts >= pert_t0) & (ts < pert_t1)
        song_pert = (song.copy(), song.copy())
        song_pert[0][mask] += rng.normal(0, 5e3, size=mask.sum())
        song_pert[1][mask] += rng.normal(0, 5e4, size=mask.sum())
        spec_song_weak_pert.append(get_spec(song_pert[0], fs, **song_kwargs))
        spec_song_strong_pert.append(get_spec(song_pert[1], fs, **song_kwargs))

        # syl spectra
        li = []
        for j in range(len(CODE)):
            syl = song[t0_[j]-t0_[0]:t1_[j]-t0_[0]]
            li.append(get_spec(syl, fs, **syl_kwargs))
        spec_syl.append(np.stack(li, axis=0))

        # syl spectra, with perrturbation
        syl = song_pert[0][t0_[pert_syl_idx]-t0_[0]:t1_[pert_syl_idx]-t0_[0]]
        spec_syl_weak_pert.append(get_spec(syl, fs, **syl_kwargs))
        syl = song_pert[1][t0_[pert_syl_idx]-t0_[0]:t1_[pert_syl_idx]-t0_[0]]
        spec_syl_strong_pert.append(get_spec(syl, fs, **syl_kwargs))

print('Found %d songs.' % len(spec_song))
# make into np arrays
# (num_samples, *, num_freq_bins, num_time_bins)
# for spec_syl, * is num_syl = 7
_ = (spec_song, spec_song_weak_pert, spec_song_strong_pert, spec_syl, 
     spec_syl_weak_pert, spec_syl_strong_pert, syl_on, syl_off, song_Ts)
_ = list(map(lambda x: np.stack(x, axis=0), _))
spec_song, spec_song_weak_pert, spec_song_strong_pert = _[:3]
spec_syl, spec_syl_weak_pert, spec_syl_strong_pert = _[3:6]
syl_on, syl_off, song_Ts = _[6:]

np.savez('../adult_songs/data.npz', 
         fs=fs, pert_syl_idx=pert_syl_idx,
         spec_song=spec_song, spec_song_pert=(spec_song_weak_pert, spec_song_strong_pert), 
         spec_syl=spec_syl, spec_syl_pert=(spec_syl_weak_pert, spec_syl_strong_pert), 
         syl_on=syl_on, syl_off=syl_off, song_Ts=song_Ts)