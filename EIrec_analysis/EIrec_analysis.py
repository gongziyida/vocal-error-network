#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../src')
import numpy as np
from tqdm import tqdm
from scipy.special import erf, erfinv
from scipy.sparse import issparse
from models import *
from utils import *
from train_funcs import *
rng = np.random.default_rng()


#### Model Training ####
NE, NI, N_syl, N_HVC_per_syl = 600, 150, 3, 3
N_HVC = N_syl * N_HVC_per_syl
peak_rate, kernel_width = 150, 20
T_rend = 600 # Each rendition
N_rend = 25 # Number of renditions
T_burn = 500 # Burning
T = T_burn + N_rend * T_rend # Total

# Syllables and time stamps
N_shared_channels = 1
syl_cov = block_sym_mat(NE, K=N_shared_channels, var=9, cov=7.5)
syl = rng.multivariate_normal(np.ones(NE), syl_cov, size=N_syl)
tsyl_start, tsyl_end, burst_ts = generate_syl_time(T, T_burn, T_rend, N_syl, N_HVC)
save_W_ts = np.round(tsyl_end[-1]).astype(int)

_ = rng.standard_normal((N_HVC, N_rend)) * 0 # Little fluctuation
rH = generate_HVC(T, burst_ts, peak_rate+_, kernel_width+_)

# (T, NE)
aud = generate_discrete_aud(T, NE, tsyl_start, tsyl_end, syl)

gen = lognormal_gen
c = 0.5
JEE0, JEI0, JIE0, JII0 = np.array([1, 1.7, 1.2, 1.8]) / 4
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.1
JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE, sparse=c<=0.5) / np.sqrt(NE)
JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI, sparse=c<=0.5) / np.sqrt(NI)
JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE, sparse=c<=0.5) / np.sqrt(NE)
JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII, sparse=c<=0.5) / np.sqrt(NI)
J0_mean = JEE0 / np.sqrt(NE) * c

rEmax, rImax, thE, thI, sE, sI = 50, 100, -4, 0, 2, 2
phiE = lambda x: rEmax/2 * (1 + erf((x - thE) / (np.sqrt(2) * sE)))
phiI = lambda x: rImax/2 * (1 + erf((x - thI) / (np.sqrt(2) * sI)))

w0_mean_E2E, w0_std_E2E, cW_E2E = 1/N_HVC, 0, 0.05
w_inh_E2E = w0_mean_E2E*cW_E2E
tauE, tauI, dt = 30, 10, 1

netEIrec = EINet(NE, NI, N_HVC, w0_mean_E2E, phiE, phiI, tauE, tauI, 
                 JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                 w_inh=w_inh_E2E, w0_std=w0_std_E2E, cW=cW_E2E)

hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-2e-1), 
                         tauW=1e5, J0_mean=J0_mean, asyn_E=10, rE_th=1)
train_res = netEIrec.sim(hE0, hI0, rH, aud, save_W_ts, T, dt, 0, **plasticity_kwargs)

#### process connectivity matrix and SVD ####
Js = []
for j in train_res[2]['JEE']:
    Js.append(np.block([[j.toarray(), -netEIrec.JEI.toarray()*2], 
                       [netEIrec.JIE.toarray(), -netEIrec.JII.toarray()*2]]))

from scipy.linalg import svd
svd_post = svd(Js[-1])


#### Test case constants ####
T_test = T_burn + T_rend
i_pert = 1
ti, tj = int(tsyl_start[i_pert,0]), int(tsyl_end[i_pert,0])+100
N_shuffle = 20


#### Helper functions ####
def disrupt_conn(svd, idx_disrupt, shuff_I=False):
    ''' Shuffle selected left singular vectors (modes) and generate the disrupted network
    svd: Output of scipy.linalg.svd
    idx_disrupt: list of mode indices
    shuff_I: if True, the entries corresponding to the inh. neurons will be shuffled;
             otherwise, only those corresponding to the exc. neurons will be shuffled.
    '''
    idx = [i for i in range(NE+NI) if i not in idx_disrupt]
    J_trunc = svd[0][:,idx] @ np.diag(svd[1][idx]) @ svd[2][idx,:]
    
    # shuffle U and V, one mode at a time
    U, V = np.zeros((NE+NI,len(idx_disrupt))), np.zeros((len(idx_disrupt),NE+NI))
    for i, j in enumerate(idx_disrupt):
        idxE, idxI = np.arange(NE), np.arange(NI)
        rng.shuffle(idxE)
        if shuff_I:
            rng.shuffle(idxI)
        idx = np.concat([idxE, idxI+NE])
        U[:,i], V[i,:] = svd[0][idx,j], svd[2][j,idx]
        
    aux = U @ np.diag(svd[1][idx_disrupt]) @ V
    J_disrupt = J_trunc + aux
    
    netEIrec_disrupt = EINet(NE, NI, N_HVC, w0_mean_E2E, phiE, phiI, tauE, tauI, 
                             JEE=J_disrupt[:NE,:NE], JEI=-J_disrupt[:NE,NE:]/2, 
                             JIE=J_disrupt[NE:,:NE], JII=-J_disrupt[NE:,NE:]/2, 
                             w_inh=w_inh_E2E, w0_std=w0_std_E2E, cW=cW_E2E)
    netEIrec_disrupt.W = netEIrec.W.copy()

    return netEIrec_disrupt, J_disrupt, J_trunc

def response(nets, var_dir, a_range, n_points):
    ''' Probe the networks' responses to perturbations
    nets: list of EINet
    var_dir: Direction of variation. Either 'song' or 'other'.
    a_range: 2-tuple containing the min and max of the scales `a` along the direction of variation
    n_points: Number of `a` to test
    '''
    bos = syl.copy()
    a_vals = np.linspace(*a_range, num=n_points)
    li = [[] for _ in range(len(nets))]
    
    hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
    hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
    pert = rng.multivariate_normal(np.zeros(NE), syl_cov)
    
    for a in a_vals:
        if var_dir == 'other':
            bos[i_pert] = syl[i_pert] + pert * a
        elif var_dir == 'song':
            bos[i_pert] = syl[i_pert].mean() + (syl[i_pert]-syl[i_pert].mean()) * a
            pert = None
        
        aud_test = generate_discrete_aud(T_test, NE, tsyl_start[:,:1], tsyl_end[:,:1], bos)
        args = (hE0, hI0, rH[:T_test], aud_test, [], T_test, dt, 0)
        for i, net in enumerate(nets):
            li[i].append(net.sim(*args, no_progress_bar=True)[0][ti:tj].mean(axis=0))
    
    return [np.stack(i, axis=0) for i in li], pert


#### Simulations ####
song_range, other_range = (-1, 2), (-1, 1)
res_onm = dict(rate=[]) # on-manifold variation
res_offm = dict(rate=[], pert=[]) # off-manifold variation
J_disr_corrs = []

for i in tqdm(range(N_shuffle)):
    net_disrupt_1, J_disrupt_1, _ = disrupt_conn(svd_post, [1])
    net_disrupt_145, J_disrupt_145, _ = disrupt_conn(svd_post, list(range(145, 155)))
    k = rng.integers(2,600)
    net_disrupt_k, J_disrupt_k, _ = disrupt_conn(svd_post, [k])
    k10 = rng.integers(1,600, size=10)
    net_disrupt_k10, J_disrupt_k10, _ = disrupt_conn(svd_post, k10)

    nets = [netEIrec, net_disrupt_1, net_disrupt_145, net_disrupt_k, net_disrupt_k10]
    r_onm, _ = response(nets, 'song', song_range, 7)
    r_offm, pert = response(nets, 'other', other_range, 7)

    res_onm['rate'].extend(r_onm)
    res_offm['rate'].extend(r_offm)
    res_offm['pert'].append(pert)
    
    # No need to save multiple J_post
    J_disrs = [J_disrupt_1, J_disrupt_145, J_disrupt_k, J_disrupt_k10]
    J_disr_corrs.extend([correlation(j[:NE,:NE], syl, dim=2) for j in J_disrs])

to_save = dict(order=['original', 'landscape', '10 memory', '1 rand', '10 rand'], 
               on_manifold=res_onm, off_manifold=res_offm, J_disr_corrs=J_disr_corrs,
               syl=syl, i_pert=i_pert, ti=ti, tj=tj, Js=Js)

import pickle
with open('../results/EIrec_J_disrupt_exp%s.pkl' % sys.argv[1], 'wb') as f:
    pickle.dump(to_save, f)