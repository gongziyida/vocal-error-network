import sys
sys.path.append('../src')
import numpy as np
from tqdm import tqdm
from scipy.special import erf, erfinv
from models import *
from utils import *
from train_funcs import *

## Preparations
rng = np.random.default_rng()
IMG_DIR = 'svg/'
RESULT_DIR = 'results/'

### Constants
NE, NI, N_HVC = 600, 150, 9
N_syl = 3
PEAK_RATE, KERNEL_WIDTH = 150, 20
tauE, tauI, dt = 30, 10, 1

### EI transfer function parameters
rEmax, rImax, thE, thI, slope = 50, 100, -4, 0, 2
phiE = lambda x: rEmax/2 * (1 + erf((x - thE) / (np.sqrt(2) * slope)))
phiI = lambda x: rImax/2 * (1 + erf((x - thI) / (np.sqrt(2) * slope)))

### FF transfer function parameters
r_rest = 2 # target rate when phi(0)
thFF = -erfinv(r_rest * 2 / rEmax - 1) * (np.sqrt(2) * slope)
phi = lambda x: rEmax/2 * (1 + erf((x - thFF) / (np.sqrt(2) * slope)))

### Constants related to time
T_rend = 600 # Each rendition
N_rend = 25 # Number of renditions
T_burn = 500 # Burning
T = T_burn + N_rend * T_rend # Total
T_test = T_burn + T_rend
N_repeat = 10

### Input related parameters
syl_cov = block_sym_mat(NE, K=1, var=9, cov=7.5)
tsyl_start, tsyl_end, burst_ts = generate_syl_time(T, T_burn, T_rend, N_syl, N_HVC)
T_song = int(tsyl_end[-1,0] - tsyl_start[0,0])

### Generate HVC activities for training
_ = np.arange(N_rend)
burst_ts = np.linspace(_*T_rend+T_burn, _*T_rend+T_burn+T_song, num=N_HVC, endpoint=False)
_ = rng.standard_normal((N_HVC, N_rend)) # Little fluctuation
rH = generate_HVC(T, burst_ts, PEAK_RATE+_*0, KERNEL_WIDTH+_*0)

### Net parameters
w0_mean, w0_std, cW_HVC2E, cW_E2E = 1/N_HVC, 0, 1, 0.05
w_inh_HVC2E = w0_mean*cW_HVC2E
w_inh_E2E = w0_mean*cW_E2E

gen = lognormal_gen
c = 0.5
JEE0, JEI0, JIE0, JII0 = np.array([1, 1.7, 1.2, 1.8]) / 4
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.1


## Experiments
df = dict(model=[], th=[], TS_similarity_pre=[], TS_similarity_post=[])

for repeat in range(N_repeat):
    syl = rng.multivariate_normal(np.ones(NE), syl_cov, size=N_syl)
    aud = generate_discrete_aud(T, NE, tsyl_start, tsyl_end, syl)
    
    JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE, sparse=True) / np.sqrt(NE)
    JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI, sparse=True) / np.sqrt(NI)
    JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE, sparse=True) / np.sqrt(NE)
    JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII, sparse=True) / np.sqrt(NI)
    
    hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
    hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
        
    for th in range(6):
        netFF = WCNet(NE, N_HVC, w0_mean, phi, tauE, 
                      w_inh=w_inh_HVC2E, w0_std=w0_std)
        netEI = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
                      JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                      w_inh=w_inh_HVC2E, w0_std=w0_std)
        netEIrec = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
                         JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                         w_inh=w_inh_E2E, w0_std=w0_std, cW=cW_E2E)
        
        res = dict(FF=[], EI=[], EIrec=[])
        plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-3e-2, 
                                 tauW=1e5, asyn_H=0, rE_th=th)
        res['FF'] = netFF.sim(hE0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), lr=dict(HVC=-3e-2), 
                                 tauW=1e5, asyn_H=0, rE_th=th)
        res['EI'] = netEI.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-6e-2), 
                                 tauW=1e5, JEE0_mean=JEE0/np.sqrt(NE), asyn_E=10, rE_th=th)
        res['EIrec'] = netEIrec.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]

        for k, v in res.items():
            df['model'].append(k)
            df['th'].append(th)
            for i, l in zip((0, N_rend-1), ('pre', 'post')):
                t0, t1 = T_burn+T_rend*i, T_burn+T_rend*i+T_song
                sim_ts = correlation(v[t0:t1], syl, dim=2, cosine=False)
                df['TS_similarity_'+l].append(sim_ts)

import pickle
with open('../results/vary_plasticity_th_toy_input.pkl', 'wb') as f:
    pickle.dump(df, f)