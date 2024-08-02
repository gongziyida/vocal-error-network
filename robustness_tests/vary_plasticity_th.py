import sys
sys.path.append('../src')
import numpy as np
from tqdm import tqdm
from scipy.special import erf, erfinv
from models import *
from utils import *
from train_funcs import *
rng = np.random.default_rng()

NE, NI, N_syl, N_HVC_per_syl = 600, 150, 3, 3
N_HVC = N_syl * N_HVC_per_syl

peak_rate, kernel_width = 150, 20

T_rend = 600 # Each rendition
N_rend = 25 # Number of renditions
T_burn = 500 # Burning
T = T_burn + N_rend * T_rend # Total

# Syllables and time stamps
# syl = rng.normal(1, 3, size=(N_syl, NE))#.clip(min=0)
N_shared_channels = 1
syl_cov = block_sym_mat(NE, K=N_shared_channels, var=9, cov=7.5)
syl = rng.multivariate_normal(np.ones(NE), syl_cov, size=N_syl)
tsyl_start, tsyl_end, burst_ts = generate_syl_time(T, T_burn, T_rend, N_syl, N_HVC)

_ = rng.standard_normal((N_HVC, N_rend)) # Little fluctuation
rH = generate_HVC(T, burst_ts, peak_rate+_, kernel_width+_)

# (T, NE)
aud = generate_discrete_aud(T, NE, tsyl_start, tsyl_end, syl)

gen = lognormal_gen
c = 0.5
JEE0, JEI0, JIE0, JII0 = np.array([1, 1.7, 1.2, 1.8]) / 4
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.1
J0_mean = JEE0 / np.sqrt(NE) * c

rEmax, rImax, thE, thI, sE, sI = 50, 100, -4, 0, 2, 2
phiE = lambda x: rEmax/2 * (1 + erf((x - thE) / (np.sqrt(2) * sE)))
phiI = lambda x: rImax/2 * (1 + erf((x - thI) / (np.sqrt(2) * sI)))

r_rest = 2 # target rate when phi(0)
rmax, s = 50, 2
th = -erfinv(r_rest * 2 / rmax - 1) * (np.sqrt(2) * s)
phi = lambda x: rmax/2 * (1 + erf((x - th) / (np.sqrt(2) * s)))

w0_mean, w0_std = 1/N_HVC, 0
cW_E2E = 0.05
w_inh_HVC2E = w0_mean
w_inh_E2E = w0_mean*cW_E2E
tauE, tauI, dt = 30, 10, 1

T_test = T_burn + T_rend
i_pert = 1
ti, tj = int(tsyl_start[i_pert,0]), int(tsyl_end[i_pert,0])

N_repeat = 10

df = dict(model=[], th=[], correct_similarity_TS=[], correct_similarity_BOS=[], 
          perturb_similarity_TS=[], perturb_similarity_BOS=[])

for th in range(6):
    for repeat in range(N_repeat):
        JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE, sparse=c<=0.5) / np.sqrt(NE)
        JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI, sparse=c<=0.5) / np.sqrt(NI)
        JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE, sparse=c<=0.5) / np.sqrt(NE)
        JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII, sparse=c<=0.5) / np.sqrt(NI)
        
        netFF = WCNet(NE, N_HVC, w0_mean, phi, tauE, 
                      w_inh=w_inh_HVC2E, w0_std=w0_std)
        netEI = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
                      JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                      w_inh=w_inh_HVC2E, w0_std=w0_std)
        netEIrec = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
                         JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                         w_inh=w_inh_E2E, w0_std=w0_std, cW=cW_E2E)
        
        hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
        hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
        plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-5e-2, 
                                 tauW=1e5, asyn_H=0, rE_th=th)
        _ = netFF.sim(hE0, rH, aud, [], T, dt, 1, **plasticity_kwargs)
        plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), lr=dict(HVC=-8e-2), 
                                 tauW=1e5, asyn_H=0, rE_th=th)
        _ = netEI.sim(hE0, hI0, rH, aud, [], T, dt, 1, **plasticity_kwargs)
        plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-2e-1), 
                                 tauW=1e5, J0_mean=J0_mean, asyn_E=10, rE_th=th)
        _ = netEIrec.sim(hE0, hI0, rH, aud, [], T, dt, 1, **plasticity_kwargs)
        
        hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
        hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
        pert_cov = block_sym_mat(NE, K=N_shared_channels, var=9, cov=7.5)
        bos = syl.copy()
        bos[i_pert] = rng.multivariate_normal(np.ones(NE), pert_cov)
        
        aud_test = generate_discrete_aud(T_test, NE, tsyl_start[:,:1], tsyl_end[:,:1], bos)

        res = dict(FF=[], EI=[], EIrec=[])
        for a in (aud[:T_test], aud_test):
            args = (rH[:T_test], a, [], T_test, dt, 1)
            res['FF'].append(netFF.sim(hE0, *args, no_progress_bar=True)[0])
            res['EI'].append(netEI.sim(hE0, hI0, *args, no_progress_bar=True)[0])
            res['EIrec'].append(netEIrec.sim(hE0, hI0, *args, no_progress_bar=True)[0])

        for k, v in res.items():
            df['model'].append(k)
            df['th'].append(th)
            for i, l in enumerate(('correct', 'perturb')):
                sim_ts = correlation(v[i][ti:tj], syl[i_pert], dim=2, cosine=True)
                sim_bos = correlation(v[i][ti:tj], bos[i_pert], dim=2, cosine=True)
                df[l+'_similarity_TS'].append(sim_ts)
                df[l+'_similarity_BOS'].append(sim_bos)

import pickle
with open('../results/vary_plasticity_th.pkl', 'wb') as f:
    pickle.dump(df, f)