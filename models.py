import numpy as np
from scipy.stats import norm
from scipy.sparse import random as srandom

class HVC_Aiv:
    def __init__(self, N_Aiv, N_HVC, c, w0_mean, w0_std, phi, tau_Aiv):
        self.N_Aiv, self.N_HVC = N_Aiv, N_HVC
        self.c, self.w0_mean, self.w0_std = c, w0_mean, w0_std
        self.phi = phi
        self.tau_Aiv = tau_Aiv

        rv = norm(loc=w0_mean, scale=w0_std)
        if c == 1:
            self.W = np.abs(rv.rvs((N_Aiv, N_HVC)))
        else:
            self.W = srandom(N_Aiv, N_HVC, c, 'csc', data_rvs=rv.rvs)
            self.W.data = np.abs(self.W.data)

    def sim(self, Aiv0, HVC_rates, aud, save_W_ts, T, dt, noise_strength,
            plasticity=None, lr=0, **plasticity_args):
        rng = np.random.default_rng()
        Aiv_rates = np.zeros((T, self.N_Aiv))
        Aiv_rates[0] = Aiv0

        Ws = [self.W.copy()]
        mean_HVC_input = np.zeros(T)

        for t in range(1, T):
            aux = self.W @ HVC_rates[t-1]
            mean_HVC_input[t-1] = aux.mean()
            noise = rng.normal(0, noise_strength, size=self.N_Aiv)
            dAiv = -Aiv_rates[t-1] + self.phi(aux + aud[t-1] + noise)
            Aiv_rates[t] = Aiv_rates[t-1] + dAiv * dt / self.tau_Aiv
            if lr != 0:
                plasticity(self.W, Aiv_rates[t], HVC_rates[t], lr, 
                           **plasticity_args)
            if t in save_W_ts:
                Ws.append(self.W.copy())
        
        return Aiv_rates, Ws, mean_HVC_input

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
    # The first dimension of syl, tsyl_start, and tsyl_end should be the same
    rng = np.random.default_rng()
    aud = np.zeros((T, N))
    for i, (ts, te) in enumerate(zip(tsyl_start, tsyl_end)):
        aud[max(0,int(np.round(ts))):min(T,int(np.round(te))),:] += syl[i]
    return aud

def correlation(sig1, sig2, g): 
    ''' 
    sig1: (T, N)
    sig2: (P, N)
    g: Function
        Applied to sig2
    '''
    sig2 = g(sig2)
    sig1 = (sig1 - sig1.mean(axis=1, keepdims=True)) / sig1.std(axis=1, keepdims=True)
    sig2 = (sig2 - sig2.mean(axis=1, keepdims=True)) / sig2.std(axis=1, keepdims=True)
    corr = sig1 @ sig2.T / sig1.shape[1]
    assert np.nanmax(np.abs(corr)) < 1 + 1e-5
    return corr