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
AUD_MAP_TYPE = sys.argv[1]
assert AUD_MAP_TYPE in ('neighbor', 'gaussian', 'discrete')

### Constants
NE, NI, N_HVC = 600, 150, 15
PEAK_RATE, KERNEL_WIDTH = 150, 20
tauE, tauI, dt = 30, 10, 1

### EI transfer function parameters
rEmax, rImax, thE, thI, slope = 50, 100, 0, 0, 2

### FF transfer function parameters
r_rest = 2 # target rate when phi(0)
thFF = -erfinv(r_rest * 2 / rEmax - 1) * (np.sqrt(2) * slope)

### Read and map auditory inputs
fname = '../realistic_auditory_processing/learned_song_responses.npz'
ma = 1/100 if AUD_MAP_TYPE=='discrete' else None
aud_real, mapping = read_realistic_input(fname, NE, mean=0, scale=2, 
                                         mapping=AUD_MAP_TYPE, mapping_args=ma)

### Time window of perturbation
PERT_T0 = int(np.round(aud_real['pert_t0'].min(), -1))
PERT_T1 = int(np.round(aud_real['pert_t1'].max(), -1)) + 100

### Constants related to time
T_post = 200 # Silence after song
T_song = aud_real['ctrl'].shape[2]
T_rend = T_song + T_post # Each rendition
N_rend = 35 # Number of renditions
T_burn = 500 # Burning
T = T_burn + N_rend * T_rend # Total
T_test = T_burn + T_rend
N_repeat = 10
template = aud_real['ctrl'].mean(axis=0).T

### Generate HVC activities for training
_ = np.arange(N_rend)
burst_ts = np.linspace(_*T_rend+T_burn, _*T_rend+T_burn+T_song, num=N_HVC, endpoint=False)
_ = rng.standard_normal((N_HVC, N_rend)) # Little fluctuation
rH = generate_HVC(T, burst_ts, PEAK_RATE+_*0, KERNEL_WIDTH+_*0)

### Net parameters
w0_mean, cW_HVC2E, cW_E2E = 0.05/N_HVC, 1, 0.05
w_inh_HVC2E = w0_mean*cW_HVC2E
w_inh_E2E = w0_mean*cW_E2E

gen = lognormal_gen
c = 0.5
srKEc, srKIc = np.sqrt(NE*c), np.sqrt(NI*c)
JEE0, JEI0, JIE0, JII0 = np.array([1/srKEc, 1.7/srKIc, 1/srKEc, 1.5/srKIc]) / 10
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.1


## Experiments
df = dict(model=[], th=[], TS_similarity_pre=[], TS_similarity_post=[])

for repeat in range(N_repeat):
    aud_real, mapping = read_realistic_input(fname, NE, mean=2, scale=3, 
                                     mapping=AUD_MAP_TYPE, mapping_args=ma)
    aud, aud_idx = generate_realistic_aud(aud_real['ctrl'], N_rend, T_burn, T_post)
    
    JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE, sparse=True)
    JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI, sparse=True)
    JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE, sparse=True)
    JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII, sparse=True)
    
    hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
    hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
        
    for th in range(6):
        netFF = WCNet(NE, N_HVC, w0_mean, (rEmax, thFF+3, slope), tauE, 
                      w0_std=0, cW=cW_HVC2E)
        netEI = EINet(NE, NI, N_HVC, w0_mean, 
                      (rEmax, thE+3, slope), (rImax, thI, slope), tauE, tauI, 
                      JEE=JEE.copy(), JEI=JEI.copy(), 
                      JIE=JIE.copy(), JII=JII.copy(), 
                      w0_std=0, cW=cW_HVC2E)
        netEIrecEE = EINet(NE, NI, N_HVC, w0_mean, 
                           (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
                           JEE=JEE.copy(), JEI=JEI.copy(), 
                           JIE=JIE.copy(), JII=JII.copy(), 
                           w0_std=0, cW=cW_E2E)
        netEIrecEI = EINet(NE, NI, N_HVC, w0_mean, 
                           (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
                           JEE=JEE.copy(), JEI=JEI.copy(), 
                           JIE=JIE.copy(), JII=JII.copy(), 
                           w0_std=0, cW=cW_E2E)
        
        res = dict(FF=[], HVC2E=[], E2E=[], E2I2E_thI5=[], E2I2E_thI10=[])
        plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-3e-2, 
                                 tauW=1e5, asyn_H=10, rE_th=th)
        res['FF'] = netFF.sim(hE0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), 
                                 lr=dict(HVC=-3e-2), tauW=1e5, asyn_H=10, rE_th=th)
        res['HVC2E'] = netEI.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-5e-2),
                                 tauW=1e5, JEE0_mean=JEE0, asyn_E=10, rE_th=th)
        res['E2E'] = netEIrecEE.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(JEI=bilin_hebb_EI,JIE=bilin_hebb_IE), 
                                 lr=dict(JEI=5e-2,JIE=5e-3), tauW=1e5, 
                                 JEI0_mean=JEI0, JIE0_mean=JIE0, 
                                 asyn_E=10, asyn_I=0, rE_th=th, rI_th=5) # rI_th = 5
        res['E2I2E_thI5'] = netEIrecEI.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]

        # redo for rI_th = 10
        netEIrecEI = EINet(NE, NI, N_HVC, w0_mean, 
                           (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
                           JEE=JEE.copy(), JEI=JEI.copy(), 
                           JIE=JIE.copy(), JII=JII.copy(), 
                           w0_std=0, cW=cW_E2E)
        plasticity_kwargs = dict(plasticity=dict(JEI=bilin_hebb_EI,JIE=bilin_hebb_IE), 
                                 lr=dict(JEI=5e-2,JIE=5e-3), tauW=1e5, 
                                 JEI0_mean=JEI0, JIE0_mean=JIE0, 
                                 asyn_E=10, asyn_I=0, rE_th=th, rI_th=10) # rI_th = 10
        res['E2I2E_thI10'] = netEIrecEI.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        
        for k, v in res.items():
            df['model'].append(k)
            df['th'].append(th)
            for i, l in zip((0, N_rend-1), ('pre', 'post')):
                t0, t1 = T_burn+T_rend*i, T_burn+T_rend*i+T_song
                sim_ts = correlation(v[t0:t1], template, dim=1, cosine=False)
                df['TS_similarity_'+l].append(sim_ts)

import pickle
with open('../results/vary_plasticity_th_realistic_input.pkl', 'wb') as f:
    pickle.dump(df, f)