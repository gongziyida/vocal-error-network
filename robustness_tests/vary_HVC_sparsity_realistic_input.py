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
rEmax, rImax, thE, thI, slope = 50, 100, -4, 0, 2
phiE = lambda x: rEmax/2 * (1 + erf((x - thE) / (np.sqrt(2) * slope)))
phiI = lambda x: rImax/2 * (1 + erf((x - thI) / (np.sqrt(2) * slope)))

### FF transfer function parameters
r_rest = 2 # target rate when phi(0)
thFF = -erfinv(r_rest * 2 / rEmax - 1) * (np.sqrt(2) * slope)
phi = lambda x: rEmax/2 * (1 + erf((x - thFF) / (np.sqrt(2) * slope)))

### Read and map auditory inputs
fname = '../realistic_auditory_processing/learned_song_responses.npz'
ma = 1/100 if AUD_MAP_TYPE=='discrete' else None
aud_real, mapping = read_realistic_input(fname, NE, mean=2, scale=3, 
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
w0_mean, w0_std = 1/N_HVC, 0

gen = lognormal_gen
c = 0.5
JEE0, JEI0, JIE0, JII0 = np.array([1, 1.7, 1.2, 1.8]) / 4
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.1

## Experiments
cWs = (0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.999) # 0.999 so all W are sparse matrices
df = dict(model=[], cW=[], TS_similarity_pre=[], TS_similarity_post=[])

for repeat in range(N_repeat):
    aud_real, mapping = read_realistic_input(fname, NE, mean=2, scale=3, 
                                     mapping=AUD_MAP_TYPE, mapping_args=ma)
    aud, aud_idx = generate_realistic_aud(aud_real['ctrl'], N_rend, T_burn, T_post)
    
    JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE, sparse=True) / np.sqrt(NE)
    JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI, sparse=True) / np.sqrt(NI)
    JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE, sparse=True) / np.sqrt(NE)
    JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII, sparse=True) / np.sqrt(NI)
    
    hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
    hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
    
    for cW in cWs:
        w_inh = w0_mean*cW
        
        netFF = WCNet(NE, N_HVC, w0_mean, phi, tauE, 
                      w_inh=w_inh, w0_std=w0_std, cW=cW)
        netEI = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
                      JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                      w_inh=w_inh, w0_std=w0_std, cW=cW)
        netEIrec = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
                         JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                         w_inh=w_inh, w0_std=w0_std, cW=cW)
        
        res = dict(FF=[], EI=[], EIrec=[])
        plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-3e-2, 
                                 tauW=1e5, asyn_H=0, rE_th=1)
        res['FF'] = netFF.sim(hE0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), lr=dict(HVC=-3e-2), 
                                 tauW=1e5, asyn_H=0, rE_th=1)
        res['EI'] = netEI.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-6e-2), 
                                 tauW=1e5, JEE0_mean=JEE0/np.sqrt(NE), asyn_E=10, rE_th=1)
        res['EIrec'] = netEIrec.sim(hE0, hI0, rH, aud, [], T, dt, 0, **plasticity_kwargs)[0]
        
        for k, v in res.items():
            df['model'].append(k)
            df['cW'].append(cW)
            for i, l in zip((0, N_rend-1), ('pre', 'post')):
                t0, t1 = T_burn+T_rend*i, T_burn+T_rend*i+T_song
                sim_ts = correlation(v[t0:t1], template, dim=1, cosine=False)
                df['TS_similarity_'+l].append(sim_ts)

import pickle
with open('../results/vary_HVC_sparsity_realistic_input.pkl', 'wb') as f:
    pickle.dump(df, f)