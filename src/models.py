import numpy as np
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
            plasticity=None, lr=0, no_progress_bar=False, **plasticity_args):
        rng = np.random.default_rng()
        rE = np.zeros((T, self.NE))
        hE = np.zeros((T, self.NE))
        rI = np.zeros((T, self.NI))
        rE[0] = rE0
        rI[0] = rI0

        Ws = [self.W.copy()]
        mean_HVC_input = np.zeros(T)

        for t in tqdm(range(1, T), disable=no_progress_bar):
            aux = self.W @ rH[t-1] - self.w_inh * rH[t-1].sum()
            mean_HVC_input[t-1] = aux.mean()
            noiseE = rng.normal(0, noise_strength, size=self.NE)
            noiseI = rng.normal(0, noise_strength, size=self.NI)
            hE[t-1] = self.JEE @ rE[t-1] - self.JEI @ rI[t-1]
            recI = self.JIE @ rE[t-1] - self.JII @ rI[t-1]
            dI = -rI[t-1] + self.phiI(recI + self.wI * rH[t-1].mean() + noiseI)
            rI[t] = rI[t-1] + dI * dt / self.tauI
            drE = -rE[t-1] + self.phiE(aux + aud[t-1] + hE[t-1] + noiseE)
            rE[t] = rE[t-1] + drE * dt / self.tauE
            if lr != 0:
                plasticity(self, rE[t], rH[t], lr, **plasticity_args)
            if t in save_W_ts:
                Ws.append(self.W.copy())
        
        return rE, rI, Ws, mean_HVC_input, hE


class EINetRecPlasticity(EINet): # Same as EI net, but with recurrent plasticity
    def sim(self, rE0, rI0, rH, aud, save_W_ts, T, dt, noise_strength, 
            plasticity=None, lr=0, asyn=0, no_progress_bar=False, **plasticity_args):
        rng = np.random.default_rng()
        rE = np.zeros((T, self.NE))
        hE = np.zeros((T, self.NE))
        rI = np.zeros((T, self.NI))
        rE[0] = rE0
        rI[0] = rI0

        Ws = [self.JEE.copy()]
        mean_HVC_input = np.zeros(T) # will not record

        for t in tqdm(range(1, T), disable=no_progress_bar):
            aux = self.W @ rH[t-1] - self.w_inh * rH[t-1].sum()
            noiseE = rng.normal(0, noise_strength, size=self.NE)
            noiseI = rng.normal(0, noise_strength, size=self.NI)
            hE[t-1] = self.JEE @ rE[t-1] - self.JEI @ rI[t-1]
            recI = self.JIE @ rE[t-1] - self.JII @ rI[t-1]
            dI = -rI[t-1] + self.phiI(recI + self.wI * rH[t-1].mean() + noiseI)
            rI[t] = rI[t-1] + dI * dt / self.tauI
            drE = -rE[t-1] + self.phiE(aux + aud[t-1] + hE[t-1] + noiseE)
            rE[t] = rE[t-1] + drE * dt / self.tauE
            if lr != 0:
                plasticity(self, rE[t], rE[max(t-asyn,0)], rH[t], lr, **plasticity_args)
            if t in save_W_ts:
                Ws.append(self.JEE.copy())
        
        return rE, rI, Ws, mean_HVC_input, hE

