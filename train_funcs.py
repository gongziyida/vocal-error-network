import numpy as np
from scipy.special import erf, erfinv
from models import *
rng = np.random.default_rng()

def generate_syl_time(T, T_burn, T_rend, N_syl, N_HVC, inter_syl_interval_ratio=0):
    # ( Number of syllables , Number of renditions )
    _ = np.arange(0, T - T_burn + T_rend, T_rend) # start and end time of each rendition
    # start and end time of each syllabus; inter-rendition interval = duration of a syllabus
    _ = np.linspace(_[:-1], _[1:], num=N_syl+1, endpoint=False) + T_burn
    tsyl_start, tsyl_end = _[:-1], _[1:]
    inter_syl_interval = (tsyl_end - tsyl_start)[0,0] * inter_syl_interval_ratio
    tsyl_start, tsyl_end = np.round(tsyl_start), np.round(tsyl_end - inter_syl_interval)
    # ( Number of HVC neurons , Number of renditions )
    burst_ts = np.linspace(tsyl_start[0,:], tsyl_end[-1,:], num=N_HVC, endpoint=True)
    return tsyl_start, tsyl_end, burst_ts
    
def generate_HVC(T, burst_ts, peak_rate, kernel_width):
    # burst_ts, peak_rate, and kernel_width must be a nested list of (N,) 
    # with inner dimension (num of bursts,)
    ts = np.arange(T)
    N = len(burst_ts)
    rates = np.zeros((T, N))
    for i in range(N):
        for bt, pr, kw in zip(burst_ts[i], peak_rate[i], kernel_width[i]):
            rates[:,i] += pr * np.exp(-(ts - bt)**2 / (2 * kw**2))
    return rates

def generate_discrete_aud(T, N, tsyl_start, tsyl_end, syl):
    N_syl = syl.shape[0]
    if len(tsyl_start.shape) == 2: # (N_syl, N_rend) or (N_rend, N_syl)
        assert tsyl_start.shape[0] == tsyl_end.shape[0]
        assert tsyl_start.shape[1] == tsyl_end.shape[1]
        if tsyl_start.shape[0] == N_syl:
            N_rend = tsyl_start.shape[1]
            tsyl_start, tsyl_end = tsyl_start.T.flatten(), tsyl_end.T.flatten()
        elif tsyl_start.shape[1] == N_syl:
            N_rend = tsyl_start.shape[0]
            tsyl_start, tsyl_end = tsyl_start.flatten(), tsyl_end.flatten()
        else:
            raise ValueError
    else:
        N_rend = int(tsyl_start.shape[0] // N_syl)
    syl = np.tile(syl, (N_rend, 1))
    
    rng = np.random.default_rng()
    aud = np.zeros((T, N))
    for i, (ts, te) in enumerate(zip(tsyl_start, tsyl_end)):
        aud[max(0,int(np.round(ts))):min(T,int(np.round(te))),:] += syl[i]
    return aud