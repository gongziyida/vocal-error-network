#!/usr/bin/env python
# coding: utf-8
import sys, pickle
sys.path.append('src')
import numpy as np
from tqdm import tqdm
from scipy.special import erf, erfinv
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
from models import *
from train_funcs import *
from visualization import *

## Preparations
rng = np.random.default_rng()
IMG_DIR = 'svg/'
RESULT_DIR = 'results/'
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

### FF transfer function parameters
r_rest = 2 # target rate when phi(0)
thFF = -erfinv(r_rest * 2 / rEmax - 1) * (np.sqrt(2) * slope)

### Read and map auditory inputs
fname = 'realistic_auditory_processing/learned_song_responses.npz'
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
# save_W_ts = np.round(burst_ts[-1]+KERNEL_WIDTH).astype(int)
save_W_ts = np.concatenate((np.linspace(0, burst_ts[-1,0], endpoint=False, num=7)[1:], 
                            np.round(burst_ts[-1]+KERNEL_WIDTH)))
save_W_ts = save_W_ts.astype(int)
save_W_trial_num = [i/8 for i in range(7)] + [i for i in range(1, N_rend + 1)]

if HVC_COND == 'mature_hvc':
    _ = rng.standard_normal((N_HVC, N_rend)) # Little fluctuation
    rH = generate_HVC(T, burst_ts, PEAK_RATE+_*0.1, KERNEL_WIDTH+_*0.01)
    
elif HVC_COND == 'developing_hvc':
    peak_rates = np.zeros_like(burst_ts)
    kernel_widths = np.zeros_like(burst_ts) + KERNEL_WIDTH
    for i in range(N_rend):
        # discount factor j; divided by np.tanh(N_rend/2/15)+1.15 to make sure j(i=0) = 1
        j = (np.tanh(-(i-N_rend/2)/15)+1.15) / (np.tanh(N_rend/2/15)+1.15)
        burst_ts[:,i] += rng.normal(loc=0, scale=100*j, size=N_HVC)
        peak_rates[:,i] = lognormal_gen(rng, PEAK_RATE, 70*j, size=N_HVC)
        kernel_widths[:,i] += rng.exponential(60*j, size=N_HVC)
    rH = generate_HVC(T, burst_ts, peak_rates, kernel_widths)
    
    # Plot HVC over training rends
    js = (0, int(np.floor(N_rend/3)), int(np.floor(N_rend*2/3)), N_rend-1)
    fig, ax = plt.subplots(1, len(js), figsize=(6, 2), sharex='all', sharey='all')
    for i, j in enumerate(js):
        ax[i].plot(rH[T_burn+j*T_rend:T_burn+(j+1)*T_rend-T_post])
        ax[i].set_title('rend%d' % (j+1), fontsize=12)
    ax[0].set(xlabel='time (ms)', ylabel='premotor rates (Hz)')
    fig.savefig(os.path.join(IMG_DIR, 'supplementary/HVC_development.svg'))
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
w0_mean_HVC2E, w0_std_HVC2E, cW_HVC2E = 0.1/N_HVC, 1e-2, 1

netFF = WCNet(NE, N_HVC, w0_mean_HVC2E, (rEmax, thFF+3, slope), tauE, w0_std=w0_std_HVC2E)
netEI = EINet(NE, NI, N_HVC, w0_mean_HVC2E, (rEmax, thE+3, slope), (rImax, thI, slope), tauE, tauI, 
              JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
              w0_std=w0_std_HVC2E, cW=cW_HVC2E)

### EI (recurrent plasticity)
w0_mean_E2E, w0_std_E2E, cW_E2E = 1/N_HVC, 0, 0.05

netEIrecEE = EINet(NE, NI, N_HVC, w0_mean_E2E, (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
                   JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                   w0_std=w0_std_E2E, cW=cW_E2E)
netEIrecEI = EINet(NE, NI, N_HVC, w0_mean_E2E, (rEmax, thE, slope), (rImax, thI, slope), tauE, tauI, 
                   JEE=JEE.copy(), JEI=JEI.copy(), JIE=JIE.copy(), JII=JII.copy(), 
                   w0_std=w0_std_E2E, cW=cW_E2E)

## Training
### Initial conditions
hE0 = rng.normal(loc=-10, scale=0.5, size=NE)
hI0 = rng.normal(loc=-1, scale=0.5, size=NI)
### Train FF
plasticity_kwargs = dict(plasticity=bilin_hebb_E_HVC, lr=-3e-2, 
                         tauW=1e5, asyn_H=10, rE_th=1.5)
rE, rI, Ws_FF, _, _ = netFF.sim(hE0, rH, aud, save_W_ts, T, dt, 1, **plasticity_kwargs)

### Train EI (HVC->E)
plasticity_kwargs = dict(plasticity=dict(HVC=bilin_hebb_E_HVC), lr=dict(HVC=-3e-2), 
                         tauW=1e5, asyn_H=10, rE_th=1.5)
rE, rI, Ws_EI, _, _ = netEI.sim(hE0, hI0, rH, aud, save_W_ts, T, dt, 1, 
                                **plasticity_kwargs)

#### Plot correlations between weights and auditory patterns
fig, ax = plot_corr_mat(Ws_FF, aud_real['ctrl'].mean(axis=0).T, 
                        aud_real['pert_strong'].mean(axis=0).T,
                        ylabel='premotor index', yticks=[5, 10], xlabel='time (ms)')
fig.savefig(os.path.join(IMG_DIR, 'supplementary/FF_learn_res_%s.svg' % HVC_COND))
fig, ax = plot_corr_mat(Ws_EI['HVC'], aud_real['ctrl'].mean(axis=0).T, 
                        aud_real['pert_strong'].mean(axis=0).T,
                        ylabel='premotor index', yticks=[5, 10], xlabel='time (ms)')
fig.savefig(os.path.join(IMG_DIR, 'supplementary/EI_learn_res_%s.svg' % HVC_COND))

#### Plot example neuron and HVC weights during training
rends = (0, 10, 25)
# Pick the one that fires at the perturbation time window
HVC_idx = np.where((burst_ts[:,0] > PERT_T0+T_burn) & (burst_ts[:,0] < PERT_T1+T_burn))[0]
_ = [rE[T_burn+j*T_rend:T_burn+(j+1)*T_rend-T_post].mean(axis=0) for j in rends]
neuron_idx = rng.choice(np.where((_[0]>_[1]+0.1)&(_[1]>_[2]+0.1))[0]) # Example neuron idx

fig, ax = plt.subplots(2, 3, figsize=(4, 2), sharey='row', sharex='row', 
                       height_ratios=[1, 2])
for c, j in enumerate(rends):
    ax[0,c].plot(rE[T_burn+j*T_rend-50:T_burn+(j+1)*T_rend-T_post,neuron_idx], color='k')
    ax[0,c].axvline(50, c='k', alpha=0.6, ls='--')
    ax[0,c].set_title('Rend. %d' % j, fontsize=10)
    ax[1,c].plot(normalize(aud_real['ctrl'][:,:,300:350].mean(axis=(0,2)), 0), 
                 normalize(Ws_EI['HVC'][j][:,HVC_idx], 0), '.', 
                 ms=2, color='k')
    ax[1,c].set(xlim=[-5, 5], ylim=[-5, 5], xticks=[-3, 0, 3])
    ax[0,c].set_axis_off()
ax[0,0].plot([-75, -75], [0, 15], c='k', lw=2)
ax[0,0].text(-150, 0, '15 Hz', va='bottom', ha='center', rotation=90)
ax[0,0].text(-400, 0, 'Exc. rate', va='bottom', ha='center', rotation=90)
ax[1,0].set(yticks=[-3, 0, 3], ylabel='premotor weights')
ax[1,1].set_xlabel('auditory input patterns of tutor song')
fig.savefig(os.path.join(IMG_DIR, 'training_res_%s.svg' % HVC_COND))

### Train EI (E->E plasticity)
plasticity_kwargs = dict(plasticity=dict(JEE=bilin_hebb_EE), lr=dict(JEE=-5e-2),
                         tauW=1e5, JEE0_mean=JEE0, asyn_E=10, rE_th=1.5)
rE, rI, Ws_EIrecEE, _, _ = netEIrecEE.sim(hE0, hI0, rH, aud, save_W_ts, T, dt, 1, 
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
rE, rI, Ws_EIrecEI, _, _ = netEIrecEI.sim(hE0, hI0, rH, aud, save_W_ts, T, dt, 1, 
                                          **plasticity_kwargs)

#### Plot correlations between weights and auditory patterns
for name, Ws_EIrec, (k, b) in zip(('EE', ALT_REC_PLASTICITY), (Ws_EIrecEE, Ws_EIrecEI), 
                                  (('JEE', 'min'), ('JEI', 'max'))):
    fig, ax = plot_corr_mat(Ws_EIrec[k], aud_real['ctrl'].mean(axis=0).T, 
                            aud_real['pert_strong'].mean(axis=0).T, sortby=b, 
                            ylabel='presyn. neuron index', xlabel='time (ms)')
    fig.savefig(os.path.join(IMG_DIR, 'EIrec_learn_res_%s.svg' % (name + HVC_COND)))

#### Plot correlations between weights and auditory patterns before and after learning
_ = aud_real['ctrl'].mean(axis=0)[:,PERT_T0:PERT_T1].T

# before and after
J_corrs = [correlation(netEI.JEE.toarray().T, _), 
           correlation(netEIrecEE.JEE.toarray().T, _)]

HVC_idx = np.where((burst_ts > PERT_T0+T_burn) & (burst_ts < PERT_T1+T_burn))[0][0] 
mask = netEIrecEE.W.toarray()[:,HVC_idx] != 0
# [[with HVC before, with HVC after], [w/o HVC before, w/o HVC after]]
aux = [[J_corrs[i][mask,:].mean(axis=1) for i in (0, 1)], 
       [J_corrs[i][~mask,:].mean(axis=1) for i in (0, 1)]]
fig, ax = plt.subplots(1, sharey='all', figsize=(3.5, 2))
for i, c in enumerate(('k', 'grey')):
    ax.boxplot(aux[i], positions=[(i-0.5)*0.4, 1+(i-0.5)*0.4],
               widths=0.3, flierprops=dict(ms=2, mec=c), 
               boxprops=dict(color=c), capprops=dict(color=c),
               whiskerprops=dict(color=c), medianprops=dict(color=c))
ax.plot(0, 0, c='k', label='with premotor input')
ax.plot(0, 0, c='grey', label='no premotor input')
ax.plot()
ax.legend(title='Neurons', alignment='left', fontsize=10, title_fontsize=8)
ax.set(xlim=[-0.5, 1.5], xticks=[0, 1], 
       xticklabels=['before learning', 'after learning'])
ax.set_title('Correlation between E$\\to$E synaptic weights\nand tutor song pattern')
def statistic(x, y):
    return x.mean() - y.mean()
y = max(ax.get_ylim())
ax.plot([-0.2, 0.2], [y, y], c='k', lw=2)
ax.plot([0.8, 1.2], [y, y], c='k', lw=2)

# want to see one-sided so *2
pvs = [permutation_test([aux[0][i], aux[1][i]], statistic).pvalue*2 for i in (0, 1)]
print(pvs)
for i, pv in enumerate(pvs):
    if pv < 0.05:
        ax.text(i, y, '*' if pv > 0.01 else '**', ha='center', c='k')
    else:
        ax.text(i, y+0.01, 'ns', ha='center', c='k')
fig.savefig(os.path.join(IMG_DIR, 'EIrec_train_result_%s.svg' % HVC_COND))


## Save results
### Save models
with open(os.path.join(RESULT_DIR, 'trained_models_%s_map_%s_%s_%s.pkl') % \
          (AUD_MAP_TYPE, ALT_REC_PLASTICITY, HVC_COND, TID), 'wb') as f:
    # need to save the mapping from sparse coding output dim to neuron dim too
    pickle.dump({'FF': netFF, 'EI-HVC2E': netEI, 'EI-E2E': netEIrecEE, 
                 'EI-E2I2E': netEIrecEI, 'mapping': mapping}, f)

### Save EIrec weights
with open(os.path.join(RESULT_DIR, 'EIrec_weights_evolve_%s_map_%s_%s_%s.pkl') % \
          (AUD_MAP_TYPE, ALT_REC_PLASTICITY, HVC_COND, TID), 'wb') as f:
    pickle.dump((save_W_trial_num, {'E2E': Ws_EIrecEE, 'E2I': Ws_EIrecEI}), f)