#!/usr/bin/env python
# coding: utf-8
import sys, pickle
sys.path.append('../src')
from tqdm import tqdm
import numpy as np
from scipy.special import erf
from scipy.linalg import svd
from models import *
from train_funcs import *
from utils import *
from RL import *
from helper_funcs import *

## Preparations
rng = np.random.default_rng()
IMG_DIR = 'svg/'
RESULT_DIR = 'results/'
REC_PLASTICITY, TID = sys.argv[1:3]
AUD_MAP_TYPE = 'neighbor'
PERT_MODE = 'shuffleAll'

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

### Generate auditory inputs and HVC firing for training
aud, _ = generate_realistic_aud(aud_real['ctrl'], N_rend, T_burn, T_post)
_ = np.arange(N_rend)
# (N_HVC, N_rend)
burst_ts = np.linspace(_*T_rend+T_burn, _*T_rend+T_burn+T_song, num=N_HVC, endpoint=False)

save_W_ts = np.round(burst_ts[-1]+KERNEL_WIDTH).astype(int)

_ = np.zeros((N_HVC, N_rend))
rH = generate_HVC(T, burst_ts, PEAK_RATE+_, KERNEL_WIDTH+_)

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
w0_mean, cW = 1/N_HVC, 0.05

net = EINet(NE, NI, N_HVC, w0_mean, (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
            JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
            w0_std=0, cW=cW)

## Training
hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
if REC_PLASTICITY == 'EE':
    plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-5e-2),
                             tauW=1e5, JEE0_mean=JEE0/np.sqrt(NE), asyn_E=10, rE_th=1.5)
elif REC_PLASTICITY == 'EI':
    plasticity_kwargs = dict(plasticity=dict(JEI=bilin_hebb_EI), lr=dict(JEI=5e-2), tauW=1e5, 
                             JEI0_mean=JEI0/np.sqrt(NI), asyn_I=10, rE_th=1.5)
elif REC_PLASTICITY == 'EIIE':
    plasticity_kwargs = dict(plasticity=dict(JEI=bilin_hebb_EI,JIE=bilin_hebb_IE), 
                             lr=dict(JEI=5e-2,JIE=6e-3), tauW=1e5, 
                             JEI0_mean=JEI0, JIE0_mean=JIE0, 
                             asyn_E=10, asyn_I=0, rE_th=1.5, rI_th=5)
    
train_res = net.sim(hE0, hI0, rH, aud, save_W_ts, T, dt, 1, **plasticity_kwargs)

## process connectivity matrix and SVD 
if REC_PLASTICITY == 'EE':
    Js = [get_J(net, JEE=wee.toarray()) for wee in train_res[2]['JEE']]
elif REC_PLASTICITY == 'EI':
    Js = [get_J(net, JEI=wei.toarray()) for wei in train_res[2]['JEI']]
elif REC_PLASTICITY == 'EIIE':
    Js = [get_J(net, JEI=wei.toarray(), JIE=wie.toarray()) 
          for wei, wie in zip(train_res[2]['JEI'], train_res[2]['JIE'])]

svds = [svd(J) for J in Js]

## Get syl patterns
from scipy.signal import find_peaks
syl = aud_real['ctrl'].mean(axis=0).T
aux = np.abs(syl).mean(axis=1)
syl = syl[find_peaks(aux)[0]]

## Find modes
mem_enc, i_memory, i_nonmem = characterize_memory(svds, syl, 'left')
k_other = np.argsort(np.diff(np.log10(svds[-1][1][120:220])))[0] + 120
i_landscape = [i for i in i_nonmem if i != 0 and i < k_other]
i_others = [i for i in i_nonmem if i >= k_other]

## Helper functions
def disrupt_conn(svds, idx_disrupt, mode, t1=-1):
    ''' Shuffle selected left singular vectors (modes) and generate the disrupted network
    svds: Sequences of output of scipy.linalg.svd
    idx_disrupt: list of mode indices
    mode: `forget` (remove syllable patterns from singular vectors) or 
          `shuffle` (shuffle singular vectors, assuming the resulting SVs are still orthogonal)
    '''
    # the unaffected modes
    idx = [i for i in range(NE+NI) if i not in idx_disrupt]
    svd_post = svds[t1]
    U, V = svd_post[0][:,idx_disrupt].copy(), svd_post[2][idx_disrupt,:].copy()
    if mode == 'forget':
        mem = syl / np.linalg.norm(syl, axis=1)[:,None]
        mem_basis = svd(mem)[2][:mem.shape[0]]
        U_mem_encode = ((mem_basis @ U[:NE])[:,None,:] * mem_basis[:,:,None]).sum(axis=0)
        U[:NE] -= U_mem_encode
        U[:NE] *= np.sqrt((1 - (U[NE:]**2).sum(axis=0)[None,:]) / (U[:NE]**2).sum(axis=0)[None,:])
        assert np.allclose(np.linalg.norm(U, axis=0), 1), np.linalg.norm(U, axis=0)
        # V_mem_encode = ((V[:,:NE] @ mem_basis.T)[:,None,:] * mem_basis.T[None,:,:]).sum(axis=-1)
        # V[:,:NE] -= V_mem_encode
        # V[:,:NE] *= np.sqrt((1 - (V[:,NE:]**2).sum(axis=1)[:,None]) / (V[:,:NE]**2).sum(axis=1)[:,None])
        # assert np.allclose(np.linalg.norm(V, axis=1), 1), np.linalg.norm(V, axis=0)
    elif mode == 'shuffleE':
        idx_shuff = np.arange(0, NE)
        rng.shuffle(idx_shuff)
        U[:NE] = U[idx_shuff]
        # V[:,:NE] = V[:,idx_shuff]
    elif mode == 'shuffleAll':
        idx_shuff = np.arange(0, NE+NI)
        rng.shuffle(idx_shuff)
        U[:] = U[idx_shuff]
        # V[:] = V[:,idx_shuff]
    elif mode == 'noise':
        U[:] = rng.normal(size=U.shape)
        U /= np.linalg.norm(U, axis=0)
    elif mode == 'zero':
        U[:] = 0
    elif mode == 'swap':
        U[:] = svd_post[0][:,-len(idx_disrupt):]
    
    J_disr = svd_post[0][:,idx] @ np.diag(svd_post[1][idx]) @ svd_post[2][idx,:] \
              + U @ np.diag(svd_post[1][idx_disrupt]) @ V

    net_disr = EINet(NE, NI, N_HVC, w0_mean, 
                        (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
                        JEE=J_disr[:NE,:NE], JEI=-J_disr[:NE,NE:], 
                        JIE=J_disr[NE:,:NE], JII=-J_disr[NE:,NE:], 
                        w0_std=0, cW=cW)
    net_disr.W = net.W.copy()

    return net_disr, J_disr


## Making RL environments
adult = dict(np.load('../adult_songs/data.npz'))
n_samples, n_syl, n_freq_bins, n_time_bins = adult['spec_syl'].shape

n_basis = 30
basis = np.zeros((n_basis, n_freq_bins*n_time_bins))
coefs = np.zeros((n_samples*n_syl, n_basis))
with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
    dl = DictionaryLearning(n_components=n_basis, alpha=0.001, fit_algorithm='cd', 
                            positive_code=True, positive_dict=True)
    aux = np.concatenate([adult['spec_syl'][:,i].reshape(n_samples, -1) 
                          for i in range(n_syl)])
    coefs = dl.fit_transform(aux)
    a_std = coefs.std(axis=0)[None,:]
    coefs/= a_std
    basis = dl.components_ * a_std.T

env = Environment(basis, n_time_bins, T_song, 
                  adult['syl_on'].mean(axis=0).astype(int), 
                  adult['syl_off'].mean(axis=0).astype(int),
                  '../realistic_auditory_processing/net_params.pkl', 
                  '../results/', 'EI-E2I2E') # ve net will be replaced

# perturb and then do RL
repeats = 3
names = ['memory', 'landscape', 'rand']
for i in range(repeats):
    net_disr_mem, J_disr_mem = disrupt_conn(svds, i_memory, mode='forget')
    k_sel = i_landscape[:15] #rng.choice(i_landscape, size=15, replace=False)
    net_disr_land, J_disr_land = disrupt_conn(svds, k_sel, mode=PERT_MODE)
    k_sel = i_others[:15] #rng.choice(i_others, size=15, replace=False)
    net_disr_ctrl, J_disr_ctrl = disrupt_conn(svds, k_sel, mode=PERT_MODE)

    for net_name, net_ in zip(names, (net_disr_mem, net_disr_land, net_disr_ctrl)):
        env.ve_net = net_
        
        agent = ActorCritic(n_syl, env.action_dim)
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            ret = RL(env, agent, optimizer, max_epochs=2500)

        fname = f'EIrec_J_disr_RL_{PERT_MODE}_{int(TID)*repeats+i}_{net_name}'
        with open(f'../results/{fname}.pkl', 'wb') as f:
            to_save = dict(ve_rate=ret['ve_rate'], advantage=ret['advantage'], 
                           songs=ret['songs'][-1])
            pickle.dump(to_save, f)