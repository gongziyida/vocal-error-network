#!/usr/bin/env python
# coding: utf-8
import sys, pickle
sys.path.append('../src')
import numpy as np
from tqdm import tqdm
from scipy.special import erf, erfinv
from scipy.stats import permutation_test
from models import *
from train_funcs import *
from utils import lognormal_gen, generate_matrix

## Preparations
rng = np.random.default_rng()
RESULT_DIR = '../results/'
TID, AUD_MAP_TYPE, ALT_REC_PLASTICITY, HVC_COND = sys.argv[1:5]
assert AUD_MAP_TYPE in ('neighbor', 'gaussian', 'discrete')
assert ALT_REC_PLASTICITY in ('EI', 'EIIE')
assert HVC_COND in ('mature_hvc', 'developing_hvc')

### Constants
NE, NI, N_HVC = 600, 150, 15
PEAK_RATE, KERNEL_WIDTH = 150, 20
tauE, tauI, dt = 30, 10, 1

### EI transfer function parameters
rEmax, rImax, thE, thI, slope = 100, 100, 0, 0, 2

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

### Generate auditory inputs for training
aud, _ = generate_realistic_aud(aud_real['ctrl'], N_rend, T_burn, T_post)

### Generate HVC activities for training
_ = np.arange(N_rend)
# (N_HVC, N_rend)
burst_ts = np.linspace(_*T_rend+T_burn, _*T_rend+T_burn+T_song, num=N_HVC, endpoint=False)
save_W_ts = np.round(burst_ts[-1]+KERNEL_WIDTH).astype(int)

if HVC_COND == 'mature_hvc':
    _ = rng.standard_normal((N_HVC, N_rend)) # Little fluctuation
    rH = generate_HVC(T, burst_ts, PEAK_RATE+_*0.1, KERNEL_WIDTH+_*0.01)
    
elif HVC_COND == 'developing_hvc':
    peak_rates = np.zeros_like(burst_ts)
    kernel_widths = np.zeros_like(burst_ts) + KERNEL_WIDTH
    for i in range(N_rend):
        # discount factor j
        # j = (np.tanh(-(i-N_rend/6)/8)+1.1) / (np.tanh(N_rend/6/8)+1.1)
        j = np.exp(-3*i/N_rend)
        burst_ts[:,i] += rng.normal(loc=0, scale=100*j, size=N_HVC)
        peak_rates[:,i] = lognormal_gen(rng, PEAK_RATE, 50*j, size=N_HVC)
        kernel_widths[:,i] += rng.exponential(60*j, size=N_HVC)
    rH = generate_HVC(T, burst_ts, peak_rates, kernel_widths)
else:
    raise NotImplementError

### Initialize recurrent weights
gen = lognormal_gen
c = 0.5
# JEE0, JEI0, JIE0, JII0 = np.array([1, 1.7, 1.3, 1.8]) / 5
srKEc, srKIc = np.sqrt(NE*c), np.sqrt(NI*c)
JEE0, JEI0, JIE0, JII0 = np.array([1/srKEc, 1.7/srKIc, 1/srKEc, 1.5/srKIc]) / 10
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.1
JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE, sparse=True)
JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI, sparse=True)
JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE, sparse=True)
JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII, sparse=True)

## Initialize networks
### FF and EI (HVC->E)
w0_mean_HVC2E, w0_std_HVC2E, cW_HVC2E = 1/N_HVC, 1e-2, 1

net_FF = WCNet(NE, N_HVC, w0_mean_HVC2E, (rEmax, thE+6, slope), tauE, w0_std=w0_std_HVC2E, cW=cW_HVC2E)
net_HVC2E = EINet(NE, NI, N_HVC, w0_mean_HVC2E, (rEmax, thE+6, slope), (rImax, thI, slope), tauE, tauI, 
                  JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                  w0_std=w0_std_HVC2E, cW=cW_HVC2E)

### EI (recurrent plasticity)
w0_mean_E2E, w0_std_E2E, cW_E2E = 1/N_HVC, 0, 0.05

net_E2E = EINet(NE, NI, N_HVC, w0_mean_E2E, (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
                JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                w0_std=w0_std_E2E, cW=cW_E2E)
net_E2I = EINet(NE, NI, N_HVC, w0_mean_E2E, (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
                JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                w0_std=w0_std_E2E, cW=cW_E2E)

## Training
### Initial conditions
hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
### Train FF
plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-3e-2, 
                         tauW=1e5, asyn_H=10, rE_th=1.5)
rE_FF, rI, Ws_FF, _, _ = net_FF.sim(hE0, rH, aud, save_W_ts, T, dt, 0.1, **plasticity_kwargs)

### Train EI (HVC->E)
plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), lr=dict(HVC=-3e-2), 
                         tauW=1e5, asyn_H=10, rE_th=1.5)
rE_HVC2E, rI, Ws_HVC2E, _, _ = net_HVC2E.sim(hE0, hI0, rH, aud, save_W_ts, T, dt, 0.1, 
                                             **plasticity_kwargs)

### Train EI (E->E plasticity)
plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-5e-2),
                         tauW=1e5, JEE0_mean=JEE0, asyn_E=10, rE_th=1.5)
rE_E2E, rI, Ws_E2E, _, _ = net_E2E.sim(hE0, hI0, rH, aud, save_W_ts, T, dt, 0.1, 
                                       **plasticity_kwargs)
    
### Train EI (E->I->E or I->E plasticity)
if ALT_REC_PLASTICITY == 'EI':
    plasticity_kwargs = dict(plasticity=dict(JEI=bilin_hebb_EI), lr=dict(JEI=5e-2), tauW=1e5, 
                             JEI0_mean=JEI0, asyn_I=10, rE_th=1.5)
elif ALT_REC_PLASTICITY == 'EIIE':
    plasticity_kwargs = dict(plasticity=dict(JEI=bilin_hebb_EI,JIE=bilin_hebb_IE), 
                             lr=dict(JEI=5e-2,JIE=6e-3), tauW=1e5, 
                             JEI0_mean=JEI0, JIE0_mean=JIE0, 
                             asyn_E=10, asyn_I=0, rE_th=1.5, rI_th=5)
rE_E2I, rI, Ws_E2I, _, _ = net_E2I.sim(hE0, hI0, rH, aud, save_W_ts, T, dt, 0.1, 
                                       **plasticity_kwargs)

## Save results
### Save models
with open(os.path.join(RESULT_DIR, 'trained_models_%s_map_%s_%s_%s.pkl') % \
          (AUD_MAP_TYPE, ALT_REC_PLASTICITY, HVC_COND, TID), 'wb') as f:
    # need to save the mapping from sparse coding output dim to neuron dim too
    pickle.dump({'FF': net_FF, 'EI-HVC2E': net_HVC2E, 'EI-E2E': net_E2E, 
                 'EI-E2I2E': net_E2I, 'mapping': mapping}, f)

### Save EIrec weights
# with open(os.path.join(RESULT_DIR, 'EIrec_weights_evolve_%s_map_%s_%s_%s.pkl') % \
#           (AUD_MAP_TYPE, ALT_REC_PLASTICITY, HVC_COND, TID), 'wb') as f:
#     pickle.dump({'E2E': Ws_E2E, 'E2I': Ws_E2I}, f)