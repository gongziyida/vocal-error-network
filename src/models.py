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

    def sim(self, hE0, rH, aud, save_W_ts, T, dt, noise_strength, 
            plasticity=None, lr=0, hI0=0, no_progress_bar=False, **plasticity_args):
        rng = np.random.default_rng()
        rE = np.zeros((T, self.NE))
        rI = np.zeros(T) # rI is already mean in WC model
        hE, hI = np.zeros(self.NE)+hE0, hI0
        rE[0] = self.phiE(hE0)
        rI[0] = self.phiE(hI0)

        Ws = [self.W.copy()]
        mean_HVC_input = np.zeros(T)

        for t in tqdm(range(1, T), disable=no_progress_bar):
            aux = self.W @ rH[t-1] - self.w_inh * rH[t-1].sum()
            mean_HVC_input[t-1] = aux.mean()
            
            noise = rng.normal(0, noise_strength, size=self.NE)
            
            recE, recI = 0, 0
            if self.JEI != 0: # recurrent; need to calc. rI
                recE = self.JEE * rE[t-1].mean() - self.JEI * rI[t-1]
                recI = self.JIE * rE[t-1].mean() - self.JII * rI[t-1]
                
                dI = -hI + recI + self.wI * rH[t-1].mean()
                hI += dI * dt / self.tauI
                rI[t] = self.phiE(hI)
                
            dE = -hE + aux + aud[t-1] + recE + noise
            hE += dE * dt / self.tauE
            rE[t] = self.phiE(hE)
            
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
        
    def sim(self, hE0, hI0, rH, aud, save_W_ts, T, dt, noise_strength, 
            plasticity=dict(), lr=dict(), no_progress_bar=False, **plasticity_args):
        rng = np.random.default_rng()
        rE = np.zeros((T, self.NE))
        recE = np.zeros((T, self.NE))
        rI = np.zeros((T, self.NI))
        hE, hI = np.zeros(self.NE) + hE0, np.zeros(self.NI) + hI0
        rE[0] = self.phiE(hE)
        rI[0] = self.phiE(hI)

        mean_HVC_input = np.zeros(T)
        Ws = dict()
        if 'HVC' in lr.keys():
            Ws['HVC'] = [self.W.copy()]
        if 'JEE' in lr.keys():
            Ws['JEE'] = [self.JEE.copy()]
        if 'JEI' in lr.keys():
            Ws['JEI'] = [self.JEI.copy()]
        if 'JIE' in lr.keys():
            Ws['JIE'] = [self.JIE.copy()]

        for t in tqdm(range(1, T), disable=no_progress_bar):
            aux = self.W @ rH[t-1] - self.w_inh * rH[t-1].sum()
            if 'HVC' in lr.keys():
                mean_HVC_input[t-1] = aux.mean()
            
            noiseE = rng.normal(0, noise_strength, size=self.NE)
            noiseI = rng.normal(0, noise_strength, size=self.NI)
            
            recE[t-1] = self.JEE @ rE[t-1] - self.JEI @ rI[t-1]
            recI = self.JIE @ rE[t-1] - self.JII @ rI[t-1]
            
            dE = -hE + aux + aud[t-1] + recE[t-1] + noiseE
            dI = -hI + recI + self.wI * rH[t-1].mean() + noiseI
            hE += dE * dt / self.tauE
            hI += dI * dt / self.tauI
            rE[t] = self.phiE(hE)
            rI[t] = self.phiE(hI)
            
            if len(plasticity) > 0:
                for k, f in plasticity.items():
                    pre = rH if k=='HVC' else rI
                    f(self, t, rE, pre, lr[k], **plasticity_args)
            if t in save_W_ts:
                if 'HVC' in lr.keys():
                    Ws['HVC'].append(self.W.copy())
                if 'JEE' in lr.keys():
                    Ws['JEE'].append(self.JEE.copy())
                if 'JEI' in lr.keys():
                    Ws['JEI'].append(self.JEI.copy())
                if 'JIE' in lr.keys():
                    Ws['JIE'].append(self.JIE.copy())
        
        return rE, rI, Ws, mean_HVC_input, recE


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
        
def bilin_hebb_EE(net, t, rE, _, lr, JEE0_mean, tauW, asyn_E, rE_th, **kwargs):
    ## lr < 0, anti-Hebbian
    rE_post, rE_pre = rE[t], rE[max(t-asyn_E,0)]
    if issparse(net.JEE):
        aux = np.zeros(net.JEE.data.shape)
        _outer(aux, rE_post - rE_th, rE_pre, net.JEE.indptr, net.JEE.indices)
        dW = lr * aux - (net.JEE.data - JEE0_mean)
        net.JEE.data = np.clip(net.JEE.data + dW / tauW, a_min=1e-10, a_max=None)
    else:
        aux = (rE_post[:,None] - rE_th) * rE_pre[None,:]
        dW = lr * aux - (net.JEE - JEE0_mean)
        net.JEE = np.clip(net.JEE + dW / tauW, a_min=1e-10, a_max=None)

def bilin_hebb_IE(net, t, rE, rI, lr, JIE0_mean, tauW, asyn_E, rE_th, rI_th, **kwargs):
    ## lr < 0, anti-Hebbian
    rI_post, rE_pre = rI[t], rE[max(t-asyn_E,0)]
    if issparse(net.JIE):
        aux = np.zeros(net.JIE.data.shape)
        _outer(aux, rI_post - rI_th, rE_pre - rE_th, net.JIE.indptr, net.JIE.indices)
        dW = lr * aux - (net.JIE.data - JIE0_mean)
        net.JIE.data = np.clip(net.JIE.data + dW / tauW, a_min=1e-10, a_max=None)
    else:
        aux = (rI_post[:,None] - rI_th) * (rE_pre[None,:] - rE_th)
        dW = lr * aux - (net.JIE - JIE0_mean)
        net.JIE = np.clip(net.JIE + dW / tauW, a_min=1e-10, a_max=None)
        
def bilin_hebb_EI(net, t, rE, rI, lr, JEI0_mean, tauW, asyn_I, rE_th, **kwargs):
    ## lr < 0, anti-Hebbian
    rE_post, rI_pre = rE[t], rI[max(t-asyn_I,0)]
    if issparse(net.JEI):
        aux = np.zeros(net.JEI.data.shape)
        _outer(aux, rE_post - rE_th, rI_pre, net.JEI.indptr, net.JEI.indices)
        dW = lr * aux - (net.JEI.data - JEI0_mean)
        net.JEI.data = np.clip(net.JEI.data + dW / tauW, a_min=1e-10, a_max=None)
    else:
        aux = (rE_post[:,None] - rE_th) * rI_pre[None,:]
        dW = lr * aux - (net.JEI - JEI0_mean)
        net.JEI = np.clip(net.JEI + dW / tauW, a_min=1e-10, a_max=None)