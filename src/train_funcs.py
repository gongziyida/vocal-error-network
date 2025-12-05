import pickle, os
import numpy as np
from scipy.special import erf, erfinv
from tqdm import tqdm
from models import *
rng = np.random.default_rng()

def generate_syl_time(T, T_burn, T_rend, N_syl, N_HVC, inter_syl_interval=0):
    ''' return matrices have shapes (N_syl, N_rend)
    '''
    _ = np.arange(0, T - T_burn + T_rend, T_rend) # start and end time of each rendition
    # start and end time of each syllabus; inter-rendition interval = duration of a syllabus
    _ = np.linspace(_[:-1], _[1:], num=N_syl+1, endpoint=False) + T_burn
    tsyl_start, tsyl_end = _[:-1], _[1:]
    tsyl_start, tsyl_end = np.round(tsyl_start), np.round(tsyl_end - inter_syl_interval)
    # ( Number of HVC neurons , Number of renditions )
    burst_ts = np.linspace(tsyl_start[0,:], tsyl_end[-1,:], num=N_HVC, endpoint=True)
    return tsyl_start, tsyl_end, burst_ts
    
def generate_HVC(T, burst_ts, peak_rate, kernel_width):
    ''' Burst_ts, peak_rate, and kernel_width must be a nested list of (N,) 
        with inner dimension (num of bursts,)
    '''
    ts = np.arange(T)
    N = len(burst_ts)
    rates = np.zeros((T, N))
    for i in range(N):
        for bt, pr, kw in zip(burst_ts[i], peak_rate[i], kernel_width[i]):
            rates[:,i] += pr * np.exp(-(ts - bt)**2 / (2 * kw**2))
    return rates

def generate_discrete_aud(T, N, tsyl_start, tsyl_end, syl):
    ''' tsyl_.* should have shape 
        (N_rend * N_syl), (N_syl, N_rend), or (N_rend, N_syl)
    '''
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
    
def generate_realistic_aud(pool, N_rend, T_burn, T_post, aud_idx=None, in_series=True):
    '''
    pool: (N_songs, NE, T_song)
    aud_idx: optional, list of indices
    in_series: if True, returned series has length T = T_burn + N_rend * (T_song + T_post).
               otherwise, return N_rend series, each has length T + T_burn + T_song + T_post.
    '''
    if aud_idx is None:
        aud_idx = rng.choice(np.arange(pool.shape[0]), size=N_rend)
    NE, T_song = pool.shape[1], pool.shape[2]

    if in_series:
        aud = np.zeros((T_burn+N_rend*(T_song+T_post), NE))
        for i in range(N_rend):
            t0 = T_burn+i*(T_song+T_post)
            aud[t0:t0+T_song] = pool[aud_idx[i]].T
    else:
        aud = np.zeros((N_rend, T_burn+T_song+T_post, NE))
        for i in range(N_rend):
            aud[i,T_burn:T_burn+T_song] = pool[aud_idx[i]].T
    return aud, aud_idx

def read_realistic_input(fname, NE, mean, scale, mapping, mapping_args):
    '''
    params
    ------
    mean, scale: the final mean and std of the input
    mapping: `neighbor` assigns locations (0 to 1) to each auditory channel i and 
                 each excitatory neuron j. The auditory input to neuron j is
                 exp(-(xi-xj)^2 / mapping_args);
             `gaussian` creates a (N_in, NE) matrix from i.i.d. standard Gaussian;
             `discrete` creates a (N_in, NE) matrix with entries = -1, 0, 1, with
                 p(-1) = p(1) = mapping_args;
             A matrix with shape (N_channels, NE)
    return
    ------
    aud_real: (N_songs, NE, T_song)
    mapping: (N_channels, NE)
    '''
    aud_real = dict(np.load(fname)) # (n_songs, N, T)
    N_in = aud_real['ctrl'].shape[1]
    if isinstance(mapping, str):
        if mapping == 'neighbor':
            N_neuron_per_input = NE // N_in
            mapping = np.zeros((N_in, NE))
            for i in range(N_in):
                mapping[i,i*N_neuron_per_input:(i+1)*N_neuron_per_input] = 1
        elif mapping == 'gaussian':
            mapping = rng.normal(0, 1, size=(N_in, NE))
        elif mapping == 'discrete':
            p = (mapping_args, 1-2*mapping_args, mapping_args)
            mapping = rng.choice((-1,0,1), size=(N_in, NE), replace=True, p=p)
    else:
        try:
            s1, s2 = mapping.shape
        except:
            raise NotImplementError
        assert (s1 == N_in) and (s2 == NE)
        
    for k in ('ctrl', 'pert'):
        # swap, project, and then swap back
        aud_real[k] = np.swapaxes(np.swapaxes(aud_real[k], 1, 2) @ mapping, 1, 2)
        aud_real[k] = aud_real[k] / aud_real[k].std() # normalize
        np.nan_to_num(aud_real[k], copy=False, nan=0)
        aud_real[k] = aud_real[k] * scale + mean
    return aud_real, mapping


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
        
        hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
        if hasattr(self.net, 'NI'):
            hI0 = rng.normal(loc=0, scale=0.5, size=self.net.NI)
            res = self.net.sim(hE0, hI0, rH, aud, [], self.T_test, self.dt, self.noise, 
                               no_progress_bar=True)[:2]
        else: # Scalar; WCNet
            res = self.net.sim(hE0, rH, aud, [], self.T_test, self.dt, self.noise, hI0=5, 
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

def load_models(folder, aud_map_type, rec_plasiticity, hvc_cond, tid):
    fname = 'trained_models_%s_map_%s_%s_%s.pkl' % \
            (aud_map_type, rec_plasiticity, hvc_cond, str(tid))
    with open(os.path.join(folder, fname), 'rb') as f:
        d = pickle.load(f)
    return d['FF'], d['EI-HVC2E'], d['EI-E2E'], d['EI-E2I2E'], d['mapping']