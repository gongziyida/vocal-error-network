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
TEST_VAR = sys.argv[2]

## Default values of variables not tested
if TEST_VAR == 'cW':
    values = (0.05, 0.15, 0.35, 0.65, 1) 
    default_vars = dict(rE_th=1.5, thE=(5, 0))
elif TEST_VAR == 'plasticity_th':
    values = (0, 1.5, 3, 4.5, 6) 
    default_vars = dict(cW=(1, 0.05), thE=(5, 0))
elif TEST_VAR == 'thE':
    values = (-2, 0, 2, 4, 6)
    default_vars = dict(rE_th=1.5, cW=(1, 0.05))
else:
    raise ValueError
    
### Constants
NE, NI, N_HVC = 600, 150, 15
PEAK_RATE, KERNEL_WIDTH = 150, 20
tauE, tauI, dt = 30, 10, 1

### EI transfer function parameters
rEmax, rImax, thI, slope = 50, 100, 0, 2

### Read and map auditory inputs
fname = '../realistic_auditory_processing/learned_song_responses.npz'
ma = 1/100 if AUD_MAP_TYPE=='discrete' else None
aud_real, mapping = read_realistic_input(fname, NE, mean=0, scale=2, 
                                         mapping=AUD_MAP_TYPE, mapping_args=ma)

### Constants related to time
T_post = 200 # Silence after song
T_song = aud_real['ctrl'].shape[2]
T_rend = T_song + T_post # Each rendition
N_rend = 35 # Number of renditions
T_burn = 500 # Burning
T = T_burn + N_rend * T_rend # Total
T_test = T_burn + T_rend
N_repeat = 5
template = aud_real['ctrl'].mean(axis=0).T

### Generate HVC activities for training
_ = np.arange(N_rend)
burst_ts = np.linspace(_*T_rend+T_burn, _*T_rend+T_burn+T_song, num=N_HVC, endpoint=False)
_ = np.zeros((N_HVC, N_rend))
rH = generate_HVC(T, burst_ts, PEAK_RATE+_, KERNEL_WIDTH+_)

### Initialize recurrent weights
gen = lognormal_gen
c = 0.5
srKEc, srKIc = np.sqrt(NE*c), np.sqrt(NI*c)
JEE0, JEI0, JIE0, JII0 = np.array([1/srKEc, 1.7/srKIc, 1/srKEc, 1.5/srKIc]) / 10
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.1

## Experiments
w0_mean = 1/N_HVC
df = dict(model=[], TS_similarity_pre=[], TS_similarity_post=[])
df[TEST_VAR] = []

for repeat in tqdm(range(N_repeat)):
    aud_real, mapping = read_realistic_input(fname, NE, mean=0, scale=2, 
                                             mapping=AUD_MAP_TYPE, mapping_args=ma)
    aud, aud_idx = generate_realistic_aud(aud_real['ctrl'], N_rend, T_burn, T_post)
    
    JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE, sparse=True)
    JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI, sparse=True)
    JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE, sparse=True)
    JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII, sparse=True)
    
    hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
    hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
    
    for val in values:
        thE_f, thE_r = (val, val) if TEST_VAR == 'thE' else default_vars['thE']
        cW_f, cW_r = (val, val) if TEST_VAR == 'cW' else default_vars['cW']
        rE_th = val if TEST_VAR == 'plasticity_th' else default_vars['rE_th']
        
        netFF = WCNet(NE, N_HVC, w0_mean, (rEmax, thE_f, slope), tauE, 
                      w0_std=0, cW=cW_f)
        netEI = EINet(NE, NI, N_HVC, w0_mean, 
                      (rEmax, thE_f, slope), (rImax, thI, slope), tauE, tauI, 
                      JEE=JEE.copy(), JEI=JEI.copy(), 
                      JIE=JIE.copy(), JII=JII.copy(), 
                      w0_std=0, cW=cW_f)
        netEIrecEE = EINet(NE, NI, N_HVC, w0_mean, 
                           (rEmax, thE_r, slope), (rImax, thI, slope), tauE, tauI, 
                           JEE=JEE.copy(), JEI=JEI.copy(), 
                           JIE=JIE.copy(), JII=JII.copy(), 
                           w0_std=0, cW=cW_r)
        netEIrecEI = EINet(NE, NI, N_HVC, w0_mean, 
                           (rEmax, thE_r, slope), (rImax, thI, slope), tauE, tauI, 
                           JEE=JEE.copy(), JEI=JEI.copy(), 
                           JIE=JIE.copy(), JII=JII.copy(), 
                           w0_std=0, cW=cW_r)
        
        res = dict()
        plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-3e-2, 
                                 tauW=1e5, asyn_H=10, rE_th=rE_th)
        res['FF'] = netFF.sim(hE0, rH, aud, [], T, dt, 0, 
                              no_progress_bar=True, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), 
                                 lr=dict(HVC=-3e-2), tauW=1e5, asyn_H=10, rE_th=rE_th)
        res['HVC2E'] = netEI.sim(hE0, hI0, rH, aud, [], T, dt, 0, 
                                 no_progress_bar=True, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-5e-2),
                                 tauW=1e5, JEE0_mean=JEE0, asyn_E=10, rE_th=rE_th)
        res['E2E'] = netEIrecEE.sim(hE0, hI0, rH, aud, [], T, dt, 0, 
                                    no_progress_bar=True, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(JEI=bilin_hebb_EI,JIE=bilin_hebb_IE), 
                                 lr=dict(JEI=5e-2,JIE=6e-3), tauW=1e5, 
                                 JEI0_mean=JEI0, JIE0_mean=JIE0, 
                                 asyn_E=10, asyn_I=0, rE_th=rE_th, rI_th=5)
        res['E2I2E'] = netEIrecEI.sim(hE0, hI0, rH, aud, [], T, dt, 0, 
                                      no_progress_bar=True, **plasticity_kwargs)[0]
        
        for k, v in res.items():
            df['model'].append(k)
            df[TEST_VAR].append(val)
            for i, l in zip((0, N_rend-1), ('pre', 'post')):
                t0, t1 = T_burn+T_rend*i, T_burn+T_rend*i+T_song
                sim_ts = correlation(v[t0:t1], template, dim=1, cosine=False)
                df['TS_similarity_'+l].append(sim_ts)

import pickle
with open('../results/robustness_exp_%s.pkl' % TEST_VAR, 'wb') as f:
    pickle.dump(df, f)