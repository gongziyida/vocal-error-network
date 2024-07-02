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
rH = generate_HVC(T, burst_ts, peak_rate+_*0.1, kernel_width+_*0.01)


def bilin_hebb(net, aiv, hvc, lr, w0_mean, tauW):
    # lr < 0, anti-Hebbian
    for i, h in enumerate(hvc):
        dW = lr * (aiv - 1) * h - (net.W[:,i] - w0_mean)
        net.W[:,i] = np.clip(net.W[:,i] + dW / tauW, a_min=0, a_max=None)

w0_mean = 1/N_HVC
w_inh, wI = w0_mean, 0.0
tauE, tauI, dt = 40, 10, 1

#### EI network ####

gen = lognormal_gen
# gen = const_gen
c = 0.5
JEE0, JEI0, JIE0, JII0 = np.array([1, 0.8, 1.2, 0.6]) / 3
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.25
# c = 1
# JEE0, JEI0, JIE0, JII0 = np.array([1, 0.8, 1.25, 0.9])
# sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.25
JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE, sparse=c<=0.5) / np.sqrt(NE)
JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI, sparse=c<=0.5) / np.sqrt(NI)
JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE, sparse=c<=0.5) / np.sqrt(NE)
JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII, sparse=c<=0.5) / np.sqrt(NI)

rEmax, rImax, thE, thI, sE, sI = 40, 100, -5, 0, 2, 2
phiE = lambda x: rEmax/2 * (1 + erf((x - thE) / (np.sqrt(2) * sE)))
phiI = lambda x: rImax/2 * (1 + erf((x - thI) / (np.sqrt(2) * sI)))

#### Feedforward ####
r_rest = 1 # target rate when phi(0)
rmax, s = 40, 2
th = -erfinv(r_rest * 2 / rmax - 1) * (np.sqrt(2) * s)
phi = lambda x: rmax/2 * (1 + erf((x - th) / (np.sqrt(2) * s)))

#### Test models ####
Ks = np.array([5, 10, 20, 40, 60, 100])

EIcorrs, FFcorrs = [], []
EIpwcorrs, FFpwcorrs = [], []
EIsparsity, FFsparsity = {_: [] for _ in (1, 2, 3)}, {_: [] for _ in (1, 2, 3)}
for K in Ks:
    EInet = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
                  JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                  w_inh=w_inh, wI=wI)
    FFnet = WCNet(NE, N_HVC, w0_mean, phi, tauE, w_inh=w_inh)
    
    syl_cov = block_sym_mat(NE, K=K, var=9, cov=8)
    syl = rng.multivariate_normal(np.ones(NE), syl_cov, size=N_syl)
    aud = generate_discrete_aud(T, NE, tsyl_start, tsyl_end, syl)
    
    rE0 = rng.normal(loc=1, scale=0.5, size=NE).clip(min=0)
    rI0 = rng.normal(loc=5, scale=0.5, size=NI).clip(min=0)

    # Training
    _ = EInet.sim(rE0, rI0, rH, aud, [], T, dt, 1, bilin_hebb,
                       lr=-8e-2, w0_mean=w0_mean, tauW=1e5)
    _ = FFnet.sim(rE0, rH, aud, [], T, dt, 1, bilin_hebb,
                       lr=-5e-2, w0_mean=w0_mean, tauW=1e5)

    # Test with perturbation
    aud_bos = aud.copy()
    for i in range(N_syl):
        for j, (t0, t1) in enumerate(zip(tsyl_start[i], tsyl_end[i])):
            aud_bos[int(t0):int(t1)] += rng.multivariate_normal(np.ones(NE), syl_cov)
    
    ret_EI = EInet.sim(rE0, rI0, rH, aud_bos, [], T, dt, 1, bilin_hebb)
    ret_FF = FFnet.sim(rE0, rH, aud_bos, [], T, dt, 1, bilin_hebb)
    
    # pairwise correlations for syl-avg responses
    EImean, FFmean = np.zeros((N_rend*N_syl, NE)), np.zeros((N_rend*N_syl, NE))
    for i in range(N_syl):
        for j, (t0, t1) in enumerate(zip(tsyl_start[i], tsyl_end[i])):
            EImean[j*N_syl+i] = ret_EI[0][int(t0):int(t1)].mean(axis=0)
            FFmean[j*N_syl+i] = ret_FF[0][int(t0):int(t1)].mean(axis=0)
    EIpwcorrs.append(correlation(EImean.T, EImean.T, dim=2))
    FFpwcorrs.append(correlation(FFmean.T, FFmean.T, dim=2))
    
    # sparsity
    for th in (1, 2, 3):
        EIsparsity[th].append((ret_EI[0][T_burn:] > th).mean(axis=1))
        FFsparsity[th].append((ret_FF[0][T_burn:] > th).mean(axis=1))

with open('../results/vary_input_dim_%s.pkl' % sys.argv[1], 'wb') as f:
    pickle.dump((EIpwcorrs, FFpwcorrs, EIsparsity, FFsparsity), f)