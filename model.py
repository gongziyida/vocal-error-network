import numpy as np
from scipy.stats import norm
from scipy.sparse import random as srandom

class HVC_Aiv:
    def __init__(self, N_Aiv, N_HVC, c, w0_mean, w0_std):
        self.N_Aiv, self.N_HVC = N_Aiv, N_HVC
        self.c, self.w0_mean, self.w0_std = c, w0_mean, w0_std

        rv = norm(loc=w0_mean, scale=w0_std)
        if c == 1:
            self.W = rv.rvs((N_Aiv, N_HVC))
        else:
            self.W = srandom(N_Aiv, N_HVC, c, 'csc', data_rvs=rv.rvs)
    
    def oja(self, lr, aiv, hvc, hvc_th=1e-2, w_max=None):
        # lr < 0, anti-Hebbian
        for i, h in enumerate(hvc):
            if h > hvc_th:
                self.W[:,i] += lr * h * (aiv - h * self.W[:,i])
                self.W[:,i] = np.clip(self.W[:,i], a_min=0, a_max=w_max)

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

def generate_discrete_aud(T, N, tsyl_start, tsyl_end, syl, noise_strength):
    # The first dimension of syl, tsyl_start, and tsyl_end should be the same
    rng = np.random.default_rng()
    aud = np.zeros((T, N)) + rng.normal(0, noise_strength, size=(T, N))
    for i, (ts, te) in enumerate(zip(tsyl_start, tsyl_end)):
        aud[max(0,int(np.round(ts))):min(T,int(np.round(te))),:] += syl[i]
    return aud