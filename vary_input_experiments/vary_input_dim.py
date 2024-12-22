#!/usr/bin/env python
# coding: utf-8
import sys, pickle
sys.path.append('../src')
import numpy as np
from scipy.special import erf, erfinv
from models import *
from utils import *
from train_funcs import *

#### Set up shared parameters and objects ####

rng = np.random.default_rng()

NE, NI, N_syl, N_HVC_per_syl = 600, 150, 3, 3
N_HVC = N_syl * N_HVC_per_syl

peak_rate, kernel_width = 150, 20

T_rend = 600 # Each rendition
N_rend = 25 # Number of renditions
T_burn = 500 # Burning
T = T_burn + N_rend * T_rend # Total

# Syllables and time stamps

tsyl_start, tsyl_end, burst_ts = generate_syl_time(T, T_burn, T_rend, N_syl, N_HVC)

_ = rng.standard_normal((N_HVC, N_rend)) # Little fluctuation
rH = generate_HVC(T, burst_ts, peak_rate+_*0, kernel_width+_*0)

w0_mean_HVC2E, w0_mean_E2E = 0.1/N_HVC, 1/N_HVC
tauE, tauI, dt = 30, 10, 1

#### EI network ####

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
gen = lognormal_gen

rEmax, rImax, thE, thI, sE, sI = 100, 100, 0, 0, 2, 2

#### Feedforward ####
r_rest = 2 # target rate when phi(0)
rmax, s = 100, 2
th = -erfinv(r_rest * 2 / rmax - 1) * (np.sqrt(2) * s)

#### Test models ####
Ks = np.array([5, 10, 20, 40, 60, 100])

FF_pwcorrs, EI_pwcorrs, EIrec_pwcorrs = [], [], []
FF_sparsity, EI_sparsity, EIrec_sparsity = [{i: [] for i in (1, 3, 5)} for _ in range(3)]
for K in Ks:
    netFF = WCNet(NE, N_HVC, w0_mean_HVC2E, (rEmax, thFF+3, slope), tauE, w0_std=0)
    netEI = EINet(NE, NI, N_HVC, w0_mean_HVC2E, (rEmax, thE+3, slope), 
                  (rImax, thI, slope), tauE, tauI, 
                  JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                  w0_std=0, cW=1)
    netEIrecEE = EINet(NE, NI, N_HVC, w0_mean_E2E, (rEmax, thE, slope), 
                       (rImax, thI, slope), tauE, tauI, 
                       JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                       w0_std=0, cW=0.05)
    netEIrecEI = EINet(NE, NI, N_HVC, w0_mean_E2E, (rEmax, thE, slope), 
                       (rImax, thI, slope), tauE, tauI, 
                       JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                       w0_std=0, cW=0.05)
    
    syl_cov = block_sym_mat(NE, K=K, var=9, cov=8)
    syl = rng.multivariate_normal(np.ones(NE), syl_cov, size=N_syl)
    aud = generate_discrete_aud(T, NE, tsyl_start, tsyl_end, syl)
    
    hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
    hI0 = rng.normal(loc=-1, scale=0.5, size=NI)

    # Training
    plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-5e-2, 
                             tauW=1e5, asyn_H=0, rE_th=1.5)
    _ = FFnet.sim(hE0, rH, aud, [], T, dt, 1, **plasticity_kwargs)
    plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), 
                             lr=dict(HVC=-8e-2), tauW=1e5, asyn_H=0, rE_th=1.5)
    _ = EInet.sim(hE0, hI0, rH, aud, [], T, dt, 1, **plasticity_kwargs)
    plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-2e-1), 
                             tauW=1e5, JEE0_mean=JEE0/np.sqrt(NE), asyn_E=10, rE_th=1.5)
    _ = EIrec.sim(hE0, hI0, rH, aud, [], T, dt, 1, **plasticity_kwargs)

    # Test with perturbation
    aud_bos = aud.copy()
    for i in range(N_syl):
        for j, (t0, t1) in enumerate(zip(tsyl_start[i], tsyl_end[i])):
            aud_bos[int(t0):int(t1)] += rng.multivariate_normal(np.ones(NE), syl_cov)
    
    FF_res = FFnet.sim(hE0, rH, aud_bos, [], T, dt, 1)
    EI_res = EInet.sim(hE0, hI0, rH, aud_bos, [], T, dt, 1)
    EIrec_res = EIrec.sim(hE0, hI0, rH, aud_bos, [], T, dt, 1)
    
    # pairwise correlations for syl-avg responses
    FF_mean, EI_mean, EIrec_mean = np.zeros((3, N_rend*N_syl, NE))
    for i in range(N_syl):
        for j, (t0, t1) in enumerate(zip(tsyl_start[i], tsyl_end[i])):
            FF_mean[j*N_syl+i] = FF_res[0][int(t0):int(t1)].mean(axis=0)
            EI_mean[j*N_syl+i] = EI_res[0][int(t0):int(t1)].mean(axis=0)
            EIrec_mean[j*N_syl+i] = EIrec_res[0][int(t0):int(t1)].mean(axis=0)
    FF_pwcorrs.append(correlation(FF_mean.T, FF_mean.T, dim=2))
    EI_pwcorrs.append(correlation(EI_mean.T, EI_mean.T, dim=2))
    EIrec_pwcorrs.append(correlation(EIrec_mean.T, EIrec_mean.T, dim=2))
    
    # sparsity
    for th in (1, 3, 5):
        FF_sparsity[th].append((FF_res[0][T_burn:] > th).mean(axis=1))
        EI_sparsity[th].append((EI_res[0][T_burn:] > th).mean(axis=1))
        EIrec_sparsity[th].append((EIrec_res[0][T_burn:] > th).mean(axis=1))

with open('../results/vary_input_dim_%s.pkl' % sys.argv[1], 'wb') as f:
    pickle.dump((FF_pwcorrs, EI_pwcorrs, EIrec_pwcorrs, 
                 FF_sparsity, EI_sparsity, EIrec_sparsity), f)