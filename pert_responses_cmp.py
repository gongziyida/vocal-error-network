#!/usr/bin/env python
# coding: utf-8
import sys, pickle
import numpy as np
from scipy.special import erf, erfinv
from models import *
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
# syl = rng.normal(1, 3, size=(N_syl, NE))#.clip(min=0)
syl_cov = np.zeros((NE,NE))
K = 1
for i in range(NE//K):
    syl_cov[K*i:K*(i+1),K*i:K*(i+1)] = 2.5
syl_cov[np.arange(NE),np.arange(NE)] = 3

syl = rng.multivariate_normal(np.ones(NE), syl_cov, size=N_syl)

tsyl_start, tsyl_end, burst_ts = generate_syl_time(T, T_burn, T_rend, N_syl, N_HVC)

_ = rng.standard_normal((N_HVC, N_rend)) # Little fluctuation
rH = generate_HVC(T, burst_ts, peak_rate+_*0.1, kernel_width+_*0.01)

aud = generate_discrete_aud(T, NE, tsyl_start, tsyl_end, syl)

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
c = 1
JEE0, JEI0, JIE0, JII0 = np.array([1, 0.8, 1.25, 0.85])
sEE, sEI, sIE, sII = np.array([JEE0, JEI0, JIE0, JII0]) * 0.3
# sEE *= 1.3
JEE = generate_matrix(NE, NE, gen, c, rng=rng, mean=JEE0, std=sEE) / np.sqrt(NE)
JEI = generate_matrix(NE, NI, gen, c, rng=rng, mean=JEI0, std=sEI) / np.sqrt(NI)
JIE = generate_matrix(NI, NE, gen, c, rng=rng, mean=JIE0, std=sIE) / np.sqrt(NE)
JII = generate_matrix(NI, NI, gen, c, rng=rng, mean=JII0, std=sII) / np.sqrt(NI)

rEmax, rImax, thE, thI, sE, sI = 40, 100, -5, 0, 2, 2
phiE = lambda x: rEmax/2 * (1 + erf((x - thE) / (np.sqrt(2) * sE)))
phiI = lambda x: rImax/2 * (1 + erf((x - thI) / (np.sqrt(2) * sI)))

EInet = EINet(NE, NI, N_HVC, w0_mean, phiE, phiI, tauE, tauI, 
              JEE=JEE, JEI=JEI, JIE=JIE, JII=JII, w_inh=w_inh, wI=wI)


rE0 = rng.normal(loc=1, scale=0.5, size=NE).clip(min=0)
rI0 = rng.normal(loc=5, scale=0.5, size=NI).clip(min=0)
_ = EInet.sim(rE0, rI0, rH, aud, [], T, dt, 1, bilin_hebb,
              lr=-8e-2, w0_mean=w0_mean, tauW=1e5)


#### Feedforward ####
r_rest = 1 # target rate when phi(0)
rmax, s = 40, 2
th = -erfinv(r_rest * 2 / rmax - 1) * (np.sqrt(2) * s)
phi = lambda x: rmax/2 * (1 + erf((x - th) / (np.sqrt(2) * s)))

FFnet = WCNet(NE, N_HVC, w0_mean, phi, tauE, w_inh=w_inh)

rE0 = r_rest
_ = FFnet.sim(rE0, rH, aud, [], T, dt, 1, bilin_hebb,
              lr=-5e-2, w0_mean=w0_mean, tauW=1e5)


#### Test models ####

EIexp = Experiment(EInet, rH, syl, noise=1, T_test=T_burn+T_rend, 
                   t_start=tsyl_start[:,:1], t_end=tsyl_end[:,:1])
FFexp = Experiment(FFnet, rH, syl, noise=1, T_test=T_burn+T_rend, 
                   t_start=tsyl_start[:,:1], t_end=tsyl_end[:,:1])

perts = []
Ks = np.array([5, 10, 50, 100, 200, 300])

for K in Ks:
    pert_mean = np.zeros(NE)
    pert_mean[:K] = 2
    pert_cov = np.zeros((NE,NE))
    pert_cov[np.arange(NE),np.arange(NE)] = 1
    pert = rng.multivariate_normal(pert_mean, pert_cov, size=N_syl)
    perts.append(pert)

pert_args = [(pert, 1) for pert in perts]
EIres = EIexp.sim_multi(['perturb']*len(Ks), [True]*len(Ks), pert_args)
FFres = FFexp.sim_multi(['perturb']*len(Ks), [True]*len(Ks), pert_args)

for i in range(len(Ks)):
    assert np.all(EIres['bos'][i] == FFres['bos'][i])

EIcorrs, FFcorrs = [], []
for i, K in enumerate(Ks):
    EIcorrs.append(correlation(EIres['rE'][i], EIres['bos'][i] - syl, dim=2))
    FFcorrs.append(correlation(FFres['rE'][i], FFres['bos'][i] - syl, dim=2))
    
EIzs, FFzs = [], []
for i, K in enumerate(Ks):
    zEI, zFF = normalize(EIres['rE'][i], axis=1), normalize(FFres['rE'][i], axis=1)
    EIzs.append((zEI[T_burn:,:K].mean(), zEI[T_burn:,K:].mean()))
    FFzs.append((zFF[T_burn:,:K].mean(), zFF[T_burn:,K:].mean()))
    
EIsparsity, FFsparsity = dict(), dict()
for th in (1, 2, 3):
    EIsparsity[th] = np.vstack(list(map(lambda x: (x[T_burn:] > th).mean(axis=1), EIres['rE'])))
    FFsparsity[th] = np.vstack(list(map(lambda x: (x[T_burn:] > th).mean(axis=1), FFres['rE'])))

with open('results/pert_cmp_%s.pkl' % sys.argv[1], 'wb') as f:
    pickle.dump((EIcorrs, FFcorrs, EIzs, FFzs, EIsparsity, FFsparsity), f)
    # pickle.dump(([_ for _ in EIres['rE']], [_ for _ in FFres['rE']], EIres['bos'], syl), f)