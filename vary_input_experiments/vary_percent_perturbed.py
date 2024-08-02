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
N_shared_channels = 5
syl_cov = block_sym_mat(NE, K=N_shared_channels, var=9, cov=7.5)

syl = rng.multivariate_normal(np.ones(NE), syl_cov, size=N_syl)

tsyl_start, tsyl_end, burst_ts = generate_syl_time(T, T_burn, T_rend, N_syl, N_HVC)

_ = rng.standard_normal((N_HVC, N_rend)) # Little fluctuation
rH = generate_HVC(T, burst_ts, peak_rate+_*0.1, kernel_width+_*0.01)

aud = generate_discrete_aud(T, NE, tsyl_start, tsyl_end, syl)

w0_mean, w0_mean_EIrec, cW_EIrec = 1/N_HVC, 1/N_HVC, 0.05
w_inh, w_inh_EIrec, wI = w0_mean, w0_mean_EIrec*cW_EIrec, 0.0
tauE, tauI, dt = 30, 10, 1


#### EI network ####

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

EInet = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
              JEE=JEE, JEI=JEI, JIE=JIE, JII=JII, w_inh=w_inh, wI=wI)

EIrec = EINet(NE, NI, N_HVC, w0_mean_EIrec, phiE, phiI, tauE, tauI, 
              JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
              w_inh=w_inh_EIrec, wI=wI, cW=cW_EIrec)

hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), 
                         lr=dict(HVC=-8e-2), tauW=1e5, asyn_H=0, rE_th=1)
_ = EInet.sim(hE0, hI0, rH, aud, [], T, dt, 1, **plasticity_kwargs)

plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-2e-1), 
                         tauW=1e5, J0_mean=J0_mean, asyn_E=10, rE_th=1)
_ = EIrec.sim(hE0, hI0, rH, aud, [], T, dt, 1, **plasticity_kwargs)


#### Feedforward ####
r_rest = 2 # target rate when phi(0)
rmax, s = 50, 2
th = -erfinv(r_rest * 2 / rmax - 1) * (np.sqrt(2) * s)
phi = lambda x: rmax/2 * (1 + erf((x - th) / (np.sqrt(2) * s)))

FFnet = WCNet(NE, N_HVC, w0_mean, phi, tauE, w_inh=w_inh)

plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-5e-2, 
                         tauW=1e5, asyn_H=0, rE_th=1)
_ = FFnet.sim(hE0, rH, aud, [], T, dt, 1, **plasticity_kwargs)


#### Test models ####
T_test = int(tsyl_end[-1,0]) # before T_burn + T_rend
FF_exp = Experiment(FFnet, rH, syl, noise=1, T_test=T_test, 
                   t_start=tsyl_start[:,:1], t_end=tsyl_end[:,:1])
EI_exp = Experiment(EInet, rH, syl, noise=1, T_test=T_test, 
                   t_start=tsyl_start[:,:1], t_end=tsyl_end[:,:1])
EIrec_exp = Experiment(EIrec, rH, syl, noise=1, T_test=T_test, 
                       t_start=tsyl_start[:,:1], t_end=tsyl_end[:,:1])

perts = []
Ks = np.array([10, 50, 100, 200, 300, 400])

for K in Ks:
    pert_mean = np.zeros(NE)
    pert_mean[:K] = 5
    pert_cov = np.zeros((NE,NE))
    pert_cov[np.arange(NE),np.arange(NE)] = 1
    pert = rng.multivariate_normal(pert_mean, pert_cov, size=N_syl)
    perts.append(pert)

pert_args = [(pert, 1) for pert in perts]
FF_res = FF_exp.sim_multi(['perturb']*len(Ks), [True]*len(Ks), pert_args)
EI_res = EI_exp.sim_multi(['perturb']*len(Ks), [True]*len(Ks), pert_args)
EIrec_res = EI_exp.sim_multi(['perturb']*len(Ks), [True]*len(Ks), pert_args)

for i in range(len(Ks)):
    assert np.all(EI_res['bos'][i] == FF_res['bos'][i])

FF_zs, EI_zs, EIrec_zs = [], [], []
for i, K in enumerate(Ks):
    # zEI, zFF = normalize(EI_res['rE'][i], axis=1), normalize(FF_res['rE'][i], axis=1)
    zFF, zEI, zEIrec = FF_res['rE'][i], EI_res['rE'][i], EIrec_res['rE'][i]
    FF_zs.append((zFF[T_burn:,:K].mean(), zFF[T_burn:,K:].mean()))
    EI_zs.append((zEI[T_burn:,:K].mean(), zEI[T_burn:,K:].mean()))
    EIrec_zs.append((zEIrec[T_burn:,:K].mean(), zEIrec[T_burn:,K:].mean()))
    
FF_sparsity, EI_sparsity, EIrec_sparsity = dict(), dict(), dict()
for th in (1, 2, 3):
    FF_sparsity[th] = np.vstack(list(map(lambda x: (x[T_burn:] > th).mean(axis=1), FF_res['rE'])))
    EI_sparsity[th] = np.vstack(list(map(lambda x: (x[T_burn:] > th).mean(axis=1), EI_res['rE'])))
    EIrec_sparsity[th] = np.vstack(list(map(lambda x: (x[T_burn:] > th).mean(axis=1), EIrec_res['rE'])))

with open('../results/vary_percent_pert_%s.pkl' % sys.argv[1], 'wb') as f:
    pickle.dump((FF_zs, EI_zs, EIrec_zs, FF_sparsity, EI_sparsity, EIrec_sparsity), f)