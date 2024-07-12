import numpy as np
from scipy.special import erf, erfinv
from tqdm import tqdm
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
    
class Experiment:
    AUD_LIST = ('correct', 'shuf_syl_idx', 'perturb', 'off')
    
    def __init__(self, net, rH, syl, noise, 
                 T_test, t_start, t_end, dt=1):
        ''' 
        noise: background noise std
        t_start, t_end: start and end of singing/playback; must be iterable
                        outside of that time range is just noise
        '''
        self.net, self.rH = net, rH[:T_test]
        self.syl, self.noise, self.T_test = syl, noise, T_test
        self.dt, self.t_start, self.t_end = dt, t_start, t_end
        self.rH_null = np.zeros_like(self.rH) # For non-singing exp
    
    def sim(self, aud, sing=True, pert_args=None, shuff_idx=None):
        ''' pert_args: if 2-tuple, (pert pattern, weight of tutor song patterns)
                       if 3-tuple, (mean, cov, weight of tutor song patterns)
            shuff_idx: list of syllable indices indicating the order of occurrence
        '''
        NE = self.net.NE # For convenience

        idx_si = np.arange(self.syl.shape[0])
        
        if aud == 'correct':
            bos = self.syl
            aud = generate_discrete_aud(self.T_test, NE, self.t_start, self.t_end, bos)
            
        elif aud == 'shuf_syl_idx':
            if shuff_idx is None:
                rng.shuffle(idx_si, axis=0)
                while (idx_si == np.arange(len(idx_si))).any():
                    rng.shuffle(idx_si, axis=0)
            else:
                idx_si = shuff_idx.copy()
            bos = self.syl.copy()[idx_si]
            aud = generate_discrete_aud(self.T_test, NE, self.t_start, self.t_end, bos)
            
        elif aud == 'perturb':
            if len(pert_args) == 2:
                pert = pert_args[0]
            elif len(pert_args) == 3:
                pert = rng.multivariate_normal(pert_args[0], pert_args[1], 
                                               size=self.syl.shape[0])
            else:
                raise NotImplementedError
            bos = self.syl * pert_args[-1] + pert
            aud = generate_discrete_aud(self.T_test, NE, self.t_start, self.t_end, bos)

        elif aud == 'off':
            bos = np.zeros_like(self.syl)
            aud = np.zeros((self.T_test, NE))

        rH = self.rH if sing else np.zeros_like(self.rH)
        
        rE0 = rng.normal(loc=1, scale=0.5, size=NE).clip(min=0)
        if hasattr(self.net, 'NI'):
            rI0 = rng.normal(loc=5, scale=0.5, size=self.net.NI).clip(min=0)
            res = self.net.sim(rE0, rI0, rH, aud, [], self.T_test, self.dt, self.noise, 
                               no_progress_bar=True)[:2]
        else: # Scalar; WCNet
            res = self.net.sim(rE0, rH, aud, [], self.T_test, self.dt, self.noise, rI0=5, 
                               no_progress_bar=True)[:2]

        return res[0], res[1], bos, idx_si

    def sim_multi(self, aud_list, sing_list, pert_args_list=None, shuff_idx_list=None):
        if pert_args_list is None:
            pert_args_list = [None] * len(aud_list)
        if shuff_idx_list is None:
            shuff_idx_list = [None] * len(aud_list)
        rEs, rIs, boses, idxs = [], [], [], []
        iterator = tqdm(zip(aud_list, sing_list, pert_args_list, shuff_idx_list), 
                        total=len(aud_list))
        for aud, sing, pert_args, shuff_idx in iterator:
            rE, rI, bos, idx = self.sim(aud, sing, pert_args, shuff_idx)
            rEs.append(rE), rIs.append(rI), boses.append(bos), idxs.append(idx)
        return dict(rE=rEs, rI=rIs, bos=boses, shuff_idx=idxs)