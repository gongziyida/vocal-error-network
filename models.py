import numpy as np
from scipy.stats import norm
from scipy.sparse import random as srandom

# A: Aiv; H: rH; I: interneuron
class AivFF:
    def __init__(self, N_A, N_H, c, w0_mean, w0_std, phi, tau_A):
        self.N_A, self.N_H = N_A, N_H
        self.c, self.w0_mean, self.w0_std = c, w0_mean, w0_std
        self.phi = phi
        self.tau_A = tau_A

        rv = norm(loc=w0_mean, scale=w0_std)
        if c == 1:
            self.W = np.abs(rv.rvs((N_A, N_H)))
        else:
            self.W = srandom(N_A, N_H, c, 'csc', data_rvs=rv.rvs)
            self.W.data = np.abs(self.W.data)

    def sim(self, rA0, rH, aud, save_W_ts, T, dt, noise_strength,
            plasticity=None, lr=0, **plasticity_args):
        rng = np.random.default_rng()
        rA = np.zeros((T, self.N_A))
        rA[0] = rA0

        Ws = [self.W.copy()]
        mean_HVC_input = np.zeros(T)

        for t in range(1, T):
            aux = self.W @ rH[t-1]
            mean_HVC_input[t-1] = aux.mean()
            noise = rng.normal(0, noise_strength, size=self.N_A)
            drA = -rA[t-1] + self.phi(aux + aud[t-1] + noise)
            rA[t] = rA[t-1] + drA * dt / self.tau_A
            if lr != 0:
                plasticity(self.W, rA[t], rH[t], lr, 
                           **plasticity_args)
            if t in save_W_ts:
                Ws.append(self.W.copy())
        
        return rA, Ws, mean_HVC_input

class AivWilsonCowan(AivFF):
    def __init__(self, N_A, N_H, c, w0_mean, w0_std, phi, tau_A,
                 tau_I, JEE, JEI, JIE, JII):
        super().__init__(N_A, N_H, c, w0_mean, w0_std, phi, tau_A)
        self.tau_I = tau_I
        self.JEE, self.JEI, self.JIE, self.JII = JEE, JEI, JIE, JII

    def sim(self, rA0, I0, rH, aud, save_W_ts, T, dt, noise_strength, ext_I=0,
            plasticity=None, lr=0, **plasticity_args):
        rng = np.random.default_rng()
        rA = np.zeros((T, self.N_A))
        rI = np.zeros(T) # rI is already mean in WC model
        rA[0] = rA0
        rI[0] = I0

        Ws = [self.W.copy()]
        mean_HVC_input = np.zeros(T)

        for t in range(1, T):
            aux = self.W @ rH[t-1]
            mean_HVC_input[t-1] = aux.mean()
            noise = rng.normal(0, noise_strength, size=self.N_A)
            rA_mean = rA[t-1].mean()
            recE = self.JEE * rA_mean - self.JEI * rI[t-1]
            recI = self.JIE * rA_mean - self.JII * rI[t-1]
            drA = -rA[t-1] + self.phi(aux + aud[t-1] + recE + noise)
            dI = -rI[t-1] + self.phi(recI + ext_I)
            rA[t] = rA[t-1] + drA * dt / self.tau_A
            rI[t] = rI[t-1] + dI * dt / self.tau_I
            if lr != 0:
                plasticity(self.W, rA[t], rH[t], lr, 
                           **plasticity_args)
            if t in save_W_ts:
                Ws.append(self.W.copy())
        
        return rA, rI, Ws, mean_HVC_input

class AivRecPlasticity(AivWilsonCowan):
    def __init__(self, N_A, N_H, c, w0_mean, w0_std, phi, tau_A,
                 tau_I, JEE, JEI, JIE, JII, wEE0_std):
        super().__init__(N_A, N_H, c, w0_mean, w0_std, phi, tau_A, 
                         tau_I, JEE, JEI, JIE, JII)
        rv = norm(loc=0, scale=wEE0_std)
        if c == 1:
            self.WEE = rv.rvs((N_A, N_A)) # abs not necessary bc. JEE
        else:
            self.WEE = srandom(N_A, N_A, c, 'csc', data_rvs=rv.rvs)

    def sim(self, rA0, I0, rH, aud, save_W_ts, T, dt, noise_strength, ext_I=0,
            plasticity_H=None, lr_H=0, plasticity_H_args=dict(),
            plasticity_EE=None, lr_EE=0, plasticity_EE_args=dict()):
        rng = np.random.default_rng()
        rA = np.zeros((T, self.N_A))
        rI = np.zeros(T)
        rA[0] = rA0
        rI[0] = I0

        Ws = [self.W.copy()]
        WEEs = [self.WEE.copy()]
        mean_HVC_input = np.zeros(T)

        for t in range(1, T):
            aux = self.W @ rH[t-1]
            mean_HVC_input[t-1] = aux.mean()
            noise = rng.normal(0, noise_strength, size=self.N_A)
            rA_mean, rI_mean = rA[t-1].mean(), rI[t-1].mean()
            rec = self.JEE * rA_mean + self.WEE @ rA[t-1] - self.JEI * rI_mean
            drA = -rA[t-1] + self.phi(aux + aud[t-1] + rec + noise)
            dI = -rI[t-1] + self.phi(self.JIE * rA_mean - self.JII * rI_mean \
                                     + ext_I)
            rA[t] = rA[t-1] + drA * dt / self.tau_A
            rI[t] = rI[t-1] + dI * dt / self.tau_I
            if lr_H != 0:
                plasticity_H(self.W, rA[t], rH[t], lr_H, **plasticity_H_args)
            if lr_EE != 0: # don't index the second arg in case of lags
                plasticity_EE(self.WEE, rA[t], rA, lr_EE, **plasticity_EE_args)
            if t in save_W_ts:
                Ws.append(self.W.copy())
                WEEs.append(self.WEE.copy())
        
        return rA, rI, Ws, WEEs, mean_HVC_input



#### Helpful functions ####

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