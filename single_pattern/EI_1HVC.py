#!/usr/bin/env python
# coding: utf-8

import sys, pickle
sys.path.append('../src')
import numpy as np
from scipy.special import erf, erfinv
from models import *
from utils import *
from train_funcs import *

print(int(sys.argv[1]))

rng = np.random.default_rng()

NE, NI, N_syl, N_HVC_per_syl = 600, 150, 1, 1
N_HVC = N_syl * N_HVC_per_syl
peak_rate, kernel_width = 150, 50
T_rend = kernel_width # Each rendition
N_rend = 100 # Number of renditions
T_burn = 500 # Burning
T = T_burn + N_rend * T_rend # Total

# Syllables and time stamps
# ( Number of syllables , Number of renditions )
_ = np.arange(0, T - T_burn + T_rend, T_rend) # start and end time of each rendition
# start and end time of each syllabus; inter-rendition interval = duration of a syllabus
_ = np.linspace(_[:-1], _[1:], num=N_syl+1, endpoint=True) + T_burn
tsyl_start, tsyl_end = _[:-1], _[1:]
tsyl_start, tsyl_end = np.round(tsyl_start), np.round(tsyl_end)

syl = rng.normal(1, 2, size=(N_syl, NE))#.clip(min=0)
syl_rand = syl.copy()
rng.shuffle(syl_rand, axis=1)

rH = np.zeros((T, N_HVC))
for i in range(N_HVC):
    bt = (tsyl_start[0] + i * kernel_width).astype(int)
    for b in bt:
        rH[b:b+kernel_width,i] = peak_rate

# (T, NE)
aud = generate_discrete_aud(T, NE, tsyl_start, tsyl_end, syl)

def test_EI(net, HVC_idx, T_test=2000, dt=1):
    T_burn = T_test // 2
    names = ['Singing\n(Correct)', 'Singing\n(Perturb)', 'Singing\n(Deafen)', 'Playback']
    ret = dict()
    
    for i, n in enumerate(names):
        rE, hI = np.zeros((T, net.NE)), np.zeros((T, net.NI))
        recE, recI = np.zeros((T, net.NE)), np.zeros((T, net.NI))
        hE = rng.normal(loc=-10, scale=0.5, size=net.NE)
        hI = rng.normal(loc=-1, scale=0.5, size=net.NI)
        rE[0] = self.phiE(hE)
        rI[0] = self.phiE(hI)
        rH, aud = np.zeros((T_test, net.NH)), np.zeros((T_test, net.NE))
        if i == 0:
            rH[T_burn:] = peak_rate
            aud[T_burn:] = syl[0]
        elif i == 1:
            rH[T_burn:] = peak_rate
            aud[T_burn:] = syl_rand[0]
        elif i == 2:
            rH[T_burn:] = peak_rate
        elif i == 3:
            aud[T_burn:] = syl[0]
    
        for t in tqdm(range(1, T_test)):
            aux = self.W @ rH[t-1] - self.w_inh * rH[t-1].sum()
            
            recE[t-1] = self.JEE @ rE[t-1] - self.JEI @ rI[t-1]
            recI = self.JIE @ rE[t-1] - self.JII @ rI[t-1]
            
            aux = net.W @ rH[t-1] - net.w_inh * rH[t-1].sum()
            recE[t-1] = net.JEE @ rE[t-1] - net.JEI @ rI[t-1]
            recI[t-1] = net.JIE @ rE[t-1] - net.JII @ rI[t-1]
            dE = -hE + aux + aud[t-1] + recE[t-1]
            dI = -hI + recI[t-1] + net.wI * rH[t-1].mean()
            hE += dE * dt / net.tauE
            hI += dI * dt / net.tauI
            rE[t] = self.phiE(hE)
            rI[t] = self.phiE(hI)
            
        aux = dict(rE=rE, rI=rI, recE=recE, recI=recI, rH=rH, aud=aud)
        ret[n] = aux
    return ret

rEmax, rImax, thE, thI, sE, sI = 50, 100, -4, 0, 2, 2
phiE = lambda x: rEmax/2 * (1 + erf((x - thE) / (np.sqrt(2) * sE)))
phiI = lambda x: rImax/2 * (1 + erf((x - thI) / (np.sqrt(2) * sI)))

def quick_net(gamma, w0_mean, w_inh, 
              JEE0=1, JEI0=1.7, JIE0=1.2, JII0=1.8, wI=0, tauE=30, tauI=10):
    gen = lognormal_gen
    # gen = const_gen
    c = 1
    sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * gamma
    # sEE *= 1.3
    JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE) / np.sqrt(NE)
    JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI) / np.sqrt(NI)
    JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE) / np.sqrt(NE)
    JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII) / np.sqrt(NI)

    net = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, w0_std=w0_mean/2,
                JEE=JEE, JEI=JEI, JIE=JIE, JII=JII, w_inh=w_inh, wI=wI)
    return net
    
w0_mean = 0.1/N_HVC
w_inh = w0_mean
net_fp = quick_net(0.1, w0_mean, w_inh)
net_lc = quick_net(0.3, w0_mean, w_inh)

plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), 
                         lr=dict(HVC=-8e-2), tauW=1e5, asyn_H=0, rE_th=1)
dt = 1
hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
hI0 = rng.normal(loc=-1, scale=0.5, size=NI)

# Fixed point scenario
fp_pre = test_EI(net_fp, [0])
train_fp = net_fp.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)
fp_post = test_EI(net_fp, [0])

# Limit cycle or multi-dimensional attractor scenario
lc_pre = test_EI(net_lc, [0])
train_lc = net_lc.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)
lc_post = test_EI(net_lc, [0])

# Save
to_save = dict()
for l1, l2 in zip(('fp_pre', 'fp_post', 'lc_pre', 'lc_post'),
                  (fp_pre, fp_post, lc_pre, lc_post)):
    T = np.where(l2['Singing\n(Correct)']['rH'][:,0] > 0)[0][0] # song onset
    to_save[l1] = {k: v['recE'][T:-1].std(axis=1) for k, v in l2.items()}
    to_save[l1]['spon'] = l2['Singing\n(Correct)']['recE'][:T].std(axis=1).mean()

with open('../results/EI_1HVC_%d.pkl' % int(sys.argv[1]), 'wb') as f:
    pickle.dump(to_save, f)