import numpy as np
from scipy.stats import norm
from scipy.sparse import random as srandom

# E: Eiv; H: HVC; I: local inhibitory interneuron
class WCNet: # Wilson-Cowan
    def __init__(self, NE, NH, w0_mean, phiE, tauE, tauI=0, phiI=None,
                 JEE=0, JEI=0, JIE=0, JII=0, w_inh=0, wI=0, w0_std=0, cW=1):
        self.NE, self.NH = NE, NH
        self.cW, self.w0_mean, self.w0_std = cW, w0_mean, w0_std
        self.phiE, self.phiI = phiE, phiI
        self.tauE, self.tauI = tauE, tauI
        self.w_inh = w_inh # Inhibition directly from HVC
        self.wI = wI # HVC to the I population

        rv = norm(loc=w0_mean, scale=w0_std)
        if cW == 1:
            self.W = np.abs(rv.rvs((NE, NH)))
        else:
            self.W = srandom(NE, NH, cW, 'csc', data_rvs=rv.rvs)
            self.W.data = np.abs(self.W.data)
            
        if np.all(JEI == 0): # np.all for compatibility (see EINet)
            print('Not a recurrent model and rI will not be calculated.')
        self.JEE, self.JEI, self.JIE, self.JII = JEE, JEI, JIE, JII

    def sim(self, rE0, rH, aud, save_W_ts, T, dt, noise_strength, 
            plasticity=None, lr=0, rI0=0, **plasticity_args):
        rng = np.random.default_rng()
        rE = np.zeros((T, self.NE))
        rI = np.zeros(T) # rI is already mean in WC model
        rE[0] = rE0
        rI[0] = rI0

        Ws = [self.W.copy()]
        mean_HVC_input = np.zeros(T)

        for t in range(1, T):
            aux = (self.W - self.w_inh) @ rH[t-1]
            mean_HVC_input[t-1] = aux.mean()
            noise = rng.normal(0, noise_strength, size=self.NE)
            recE, recI = 0, 0
            if self.JEI != 0: # recurrent; need to calc. rI
                recE = self.JEE * rE[t-1].mean() - self.JEI * rI[t-1]
                recI = self.JIE * rE[t-1].mean() - self.JII * rI[t-1]
                dI = -rI[t-1] + self.phiI(recI + self.wI * rH[t-1].mean())
                rI[t] = rI[t-1] + dI * dt / self.tauI
            drE = -rE[t-1] + self.phiE(aux + aud[t-1] + recE + noise)
            rE[t] = rE[t-1] + drE * dt / self.tauE
            if lr != 0:
                plasticity(self, rE[t], rH[t], lr, **plasticity_args)
            if t in save_W_ts:
                Ws.append(self.W.copy())
        
        return rE, rI, Ws, mean_HVC_input
        
class EINet(WCNet):
    def __init__(self, NE, NI, NH, w0_mean, phiE, phiI, tauE, tauI,
                 JEE, JEI, JIE, JII, w_inh=0, wI=0, w0_std=0, cW=1):
        self.NI = NI
        super().__init__(NE, NH, w0_mean, phiE, tauE, tauI, phiI,
                         JEE, JEI, JIE, JII, w_inh, wI, w0_std, cW)
        
    def sim(self, rE0, rI0, rH, aud, save_W_ts, T, dt, noise_strength, 
            plasticity=None, lr=0, **plasticity_args):
        rng = np.random.default_rng()
        rE = np.zeros((T, self.NE))
        rI = np.zeros((T, self.NI))
        rE[0] = rE0
        rI[0] = rI0

        Ws = [self.W.copy()]
        mean_HVC_input = np.zeros(T)

        for t in range(1, T):
            aux = (self.W - self.w_inh) @ rH[t-1]
            mean_HVC_input[t-1] = aux.mean()
            noiseE = rng.normal(0, noise_strength, size=self.NE)
            noiseI = rng.normal(0, noise_strength, size=self.NI)
            recE = self.JEE @ rE[t-1] - self.JEI @ rI[t-1]
            recI = self.JIE @ rE[t-1] - self.JII @ rI[t-1]
            dI = -rI[t-1] + self.phiI(recI + self.wI * rH[t-1].mean() + noiseI)
            rI[t] = rI[t-1] + dI * dt / self.tauI
            drE = -rE[t-1] + self.phiE(aux + aud[t-1] + recE + noiseE)
            rE[t] = rE[t-1] + drE * dt / self.tauE
            if lr != 0:
                plasticity(self, rE[t], rH[t], lr, **plasticity_args)
            if t in save_W_ts:
                Ws.append(self.W.copy())
        
        return rE, rI, Ws, mean_HVC_input


#### Helpful functions ####

def generate_matrix(dim1, dim2, rand_gen, c=1, sparse=False):
    if c < 1:
        M = srandom(dim1, dim2, c, 'csc')
        M.data[:] = rand_gen(size=len(M.data))
        if not sparse:
            M = M.toarray()
    else:
        M = rand_gen(size=(dim1, dim2))
    return M
    
def lognormal_gen(rng, mean, std):
    mean_norm = np.log(mean**2 / np.sqrt(mean**2 + std**2))
    std_norm = np.log(1 + std**2 / mean**2)
    return lambda size: rng.lognormal(mean_norm, std_norm, size=size)

def const_gen(rng, val, _=None):
    return lambda size: np.zeros(size) + val

def normalize(sig, axis):
    m = sig.mean(axis=axis, keepdims=True)
    s = sig.std(axis=axis, keepdims=True)
    return (sig - m) / s

def correlation(sig1, sig2, dim=2): 
    ''' 
    sig1: (T1, T2, ..., Tk, N)
    sig2: (P, N) if dim == 2, or (T1, T2, ..., Tk, N) if dim == 1
    dim: int
        If 2, calculate corr(sig1[t], sig2[p]) and return (T, P)
        If 1, calculate corr(sig1[t], sig2[t]) and return (T1, T2, ..., Tk)
    '''
    sig1 = normalize(sig1, -1)
    sig2 = normalize(sig2, -1)
    if dim == 1:
        corr = (sig1 * sig2).mean(axis=-1)
    elif dim == 2:
        corr = sig1 @ sig2.T / sig1.shape[-1]
    assert np.nanmax(np.abs(corr)) < 1 + 1e-5
    return corr