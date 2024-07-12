import numpy as np
from numba import njit, void, f8, i4
from scipy.stats import norm
from scipy.sparse import random as srandom
from scipy.sparse import issparse
from tqdm import tqdm

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

        if not issparse(JEI): # otherwise assume recurrent conn.
            if np.all(JEI == 0): # np.all for compatibility (see EINet)
                print('Not a recurrent model and rI will not be calculated.')
        self.JEE, self.JEI, self.JIE, self.JII = JEE, JEI, JIE, JII

    def sim(self, rE0, rH, aud, save_W_ts, T, dt, noise_strength, 
            plasticity=None, lr=0, rI0=0, no_progress_bar=False, **plasticity_args):
        rng = np.random.default_rng()
        rE = np.zeros((T, self.NE))
        rI = np.zeros(T) # rI is already mean in WC model
        rE[0] = rE0
        rI[0] = rI0

        Ws = [self.W.copy()]
        mean_HVC_input = np.zeros(T)

        for t in tqdm(range(1, T), disable=no_progress_bar):
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
                plasticity(self, t, rE, rH, lr, **plasticity_args)
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
            plasticity=dict(), lr=dict(), no_progress_bar=False, **plasticity_args):
        rng = np.random.default_rng()
        rE = np.zeros((T, self.NE))
        hE = np.zeros((T, self.NE))
        rI = np.zeros((T, self.NI))
        rE[0] = rE0
        rI[0] = rI0

        mean_HVC_input = np.zeros(T)
        Ws = dict()
        if 'HVC' in lr.keys():
            Ws['HVC'] = [self.W.copy()]
        if 'JEE' in lr.keys():
            Ws['JEE'] = [self.JEE.copy()]

        for t in tqdm(range(1, T), disable=no_progress_bar):
            aux = self.W @ rH[t-1] - self.w_inh * rH[t-1].sum()
            if 'HVC' in lr.keys():
                mean_HVC_input[t-1] = aux.mean()
            
            noiseE = rng.normal(0, noise_strength, size=self.NE)
            noiseI = rng.normal(0, noise_strength, size=self.NI)
            
            hE[t-1] = self.JEE @ rE[t-1] - self.JEI @ rI[t-1]
            recI = self.JIE @ rE[t-1] - self.JII @ rI[t-1]
            
            dI = -rI[t-1] + self.phiI(recI + self.wI * rH[t-1].mean() + noiseI)
            rI[t] = rI[t-1] + dI * dt / self.tauI
            drE = -rE[t-1] + self.phiE(aux + aud[t-1] + hE[t-1] + noiseE)
            rE[t] = rE[t-1] + drE * dt / self.tauE
            
            if len(plasticity) > 0:
                for k, f in plasticity.items():
                    f(self, t, rE, rH, lr[k], **plasticity_args)
            if t in save_W_ts:
                if 'HVC' in lr.keys():
                    Ws['HVC'].append(self.W.copy())
                if 'JEE' in lr.keys():
                    Ws['JEE'].append(self.JEE.copy())
        
        return rE, rI, Ws, mean_HVC_input, hE


#### Plasticity functions ####
@njit(void(f8[:], f8[:], f8[:], i4[:], i4[:]))
def _outer(out, post, pre, indptr, indices):
    for i in range(len(pre)):
        # range in .data for the i-th col
        p, q = indptr[i], indptr[i+1] 
        out[p:q] = post[indices[p:q]] * pre[i]

def bilin_hebb_E_HVC(net, t, rE, rH, lr, tauW, asyn_H, rE_th, **kwargs):
    ## lr < 0, anti-Hebbian
    rH = rH[max(t-asyn_H,0)]
    if issparse(net.W):
        aux = np.zeros(net.W.data.shape)
        _outer(aux, rE[t]-rE_th, rH, net.W.indptr, net.W.indices)
        dW = lr * aux - (net.W.data - net.w0_mean)
        net.W.data = np.clip(net.W.data + dW / tauW, a_min=1e-10, a_max=None)
    else:
        dW = lr * (rE[t,:,None] - rE_th) * rH[None,:] - (net.W - net.w0_mean)
        net.W = np.clip(net.W + dW / tauW, a_min=1e-10, a_max=None)
        
def bilin_hebb_EE(net, t, rE, _, lr, J0_mean, tauW, asyn_E, rE_th, **kwargs):
    ## lr < 0, anti-Hebbian
    rE_post, rE_pre = rE[t], rE[max(t-asyn_E,0)]
    if issparse(net.JEE):
        aux = np.zeros(net.JEE.data.shape)
        _outer(aux, rE_post - rE_th, rE_pre, net.JEE.indptr, net.JEE.indices)
        dW = lr * aux - (net.JEE.data - J0_mean)
        net.JEE.data = np.clip(net.JEE.data + dW / tauW, a_min=1e-10, a_max=None)
    else:
        aux = (rE_post[:,None] - rE_th) * rE_pre[None,:]
        dW = lr * aux - (net.JEE - J0_mean)
        net.JEE = np.clip(net.JEE + dW / tauW, a_min=1e-10, a_max=None)
