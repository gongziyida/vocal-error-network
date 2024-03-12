import numpy as np
import matplotlib.pyplot as plt
from models import *
from matplotlib.colors import TwoSlopeNorm

def plot_wcol_corr(W_syl_corrs, figdim, figsize):
    N_HVC, N_syl = W_syl_corrs[0].shape
    xticks = np.arange(N_syl)
    xticklabels = list(map(chr, range(65, 65+N_syl))) # 65 is A and 97 is a
    fig, ax = plt.subplots(*figdim, sharex='all', sharey='all', figsize=figsize)
    ax = ax.flatten()
    l = min(len(W_syl_corrs), len(ax))
    idx = np.round(np.linspace(0, len(W_syl_corrs)-1, num=l, endpoint=True)).astype('int')
    for i, j in enumerate(idx):
        im = ax[i].imshow(W_syl_corrs[j], vmax=1, vmin=-1, cmap='seismic', aspect='auto')
        ax[i].set_title('Rendition %d' % j, fontsize=10)
        ax[i].set(xticks=xticks, xticklabels=xticklabels)
    ax[0].set(ylabel='HVC index', yticks=[0, N_HVC-1], yticklabels=[1, N_HVC])
    
    fig.suptitle(r'Corr(col$_i$W , $\vec\xi_j$)')
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig, ax

def plot_train_stats(Ws, rE, mean_HVC_input, save_W_ts, rI=None):
    W_norms = np.array([np.linalg.norm(_, ord='fro') for _ in Ws]) # Frobenius norm
    n = 3 if rI is None else 4
    fig, ax = plt.subplots(n, sharex='all', figsize=(4, n*1.5))
    ax[0].plot(np.hstack([[0], save_W_ts]), W_norms)
    ax[1].plot(mean_HVC_input)
    ax[2].plot(rE.mean(axis=1))
    if rI is not None:
        ax[3].plot(rI if len(rI.shape) == 1 else rI.mean(axis=1))
        ax[3].set(ylabel='Mean inh. rate')
    ax[0].set(ylabel=r'$\left\langle \| W \|_F \right\rangle$')
    ax[1].set(ylabel='mean input\nfrom HVC')
    ax[2].set(ylabel='Mean exc. rate')
    ax[-1].set(xlabel='Time (a.u.)')
    return fig, ax

def plot_train_converge(Ws, W_syl_corrs, save_W_ts, rE, r_target):
    fig, ax = plt.subplots(3, sharex='all', figsize=(4, 4.5))
    dW = [np.abs(Ws[i]-Ws[i-1]).mean(axis=0) for i in range(1,len(Ws))]
    ax[0].plot(save_W_ts, dW)
    dcorr = [np.abs(W_syl_corrs[i]-W_syl_corrs[i-1]).mean(axis=0) 
             for i in range(1,len(W_syl_corrs))]
    ax[1].plot(save_W_ts, dcorr, 
               label=['syl. %d' % (i+1) for i in range(len(dcorr[0]))])
    ax[1].legend(ncols=2)
    ax[2].plot(np.abs(rE - r_target).mean(axis=1))
    ax[0].set(ylabel=r'$\left\langle |\Delta W_{ij}|\right\rangle_i$', yscale='log')
    ax[1].set(ylabel='weight corr.\nabs. change', yscale='log')
    ax[2].set(ylabel=r'$\left\langle |r^E - r^E_0| \right\rangle$', yscale='log')
    return fig, ax


def plot_tests_mean(tests, test_names, ti, tj, plot_inh=True):
    fig, ax = plt.subplots(1, len(tests), sharey='row', sharex='all', 
                           figsize=(1.25*len(tests), 2))
    for i, (test, l) in enumerate(zip(tests, test_names)):
        ax[i].plot(test[0][ti:tj].mean(axis=1)[:])
        if plot_inh:
            aux = test[1][ti:tj, ..., None]
            ax[i].plot(aux.mean(axis=1)[:])
        ax[i].set_title(l, fontsize=10)
        ax[i].set(xlabel='Time (a.u.)')
    ax[0].set(ylabel='Mean rate')
    fig.tight_layout()
    return fig, ax

def plot_tests_corrs(tests, syl_tests, syl, test_names, ti, tj, tid_perturb_input,
                     syl_order=dict(), y=0.9):
    '''
    tests: list of list containing the excitatory and inhibitory rates
    syl_tests: sylabi used for tests
    syl: control case
    ti, tj: the start and end time to plot
    tid_perturb_input: the index of the tests with perturbed aud input
    syl_order: a dictionary {test_index: [(syl_index, t_start, t_end)...]}. 
        If given, plot horizontal bars indicate the onset of each syllabus.
    y: the vertical offset of the horizontal bars
    '''
    cmap = plt.get_cmap('plasma')
    fig, ax = plt.subplots(2, len(tests), sharey='row', sharex='all', 
                           figsize=(1.25*len(tests), 3))
    for i, (test, syl_, l) in enumerate(zip(tests, syl_tests, test_names)):
        corr = correlation(test[0][ti:tj], syl, dim=2)
        ax[0,i].set_title(l, fontsize=10)
        ax[-1,i].set(xlabel='Time (a.u.)')
        for j in range(corr.shape[1]):
            ax[0,i].plot(corr[:,j], c=cmap(j/corr.shape[1]))
        if i in tid_perturb_input: # non-zero error; calc corr
            if syl_ is None: # deafen
                syl_ = 0
            corr = correlation(test[0][ti:tj], syl_ - syl, dim=2)
            for j in range(corr.shape[1]):
                ax[1,i].plot(corr[:,j], c=cmap(j/corr.shape[1]))

    for k, v in syl_order.items():
        for (i, t0, t1) in v:
            ax[0,k].plot([t0, t1], [y, y], color=cmap(i/len(v)), lw=3)
            ax[0,k].text(t0, y - 0.1, chr(65+i), color=cmap(i/len(v)), va='top')
            
    ax[0,0].set(ylabel=r'corr$(r^E, \xi)$')
    ax[1,0].set(ylabel=r'corr$(r^E, y - \xi)$')
    fig.tight_layout()
    return fig, ax
    
def plot_tests_raster(tests, test_names, ti, tj, T_burn, 
                      plot_inh=False, syl_order=dict()):
    '''
    syl_order: a dictionary {test_index: [(syl_index, t_start, t_end)...]}. 
        If given, plot horizontal bars indicate the onset of each syllabus.
    '''
    pop = ('E', 'I') if plot_inh else ('E',)
    fig, ax = plt.subplots(len(pop), len(tests)+1, 
                           figsize=(1.25*len(tests), 2*len(pop)), 
                           width_ratios=[1]*len(tests)+[0.05])
    if not plot_inh:
        ax = ax[None,:]
    for p in range(len(pop)):
        zs = []
        for k, l in enumerate(tests):
            zs.append(normalize(l[p][ti:tj], axis=0))

        zmin = max(min(list(map(lambda _: _[0].min(), zs))), -5)
        zmax = min(max(list(map(lambda _: _[0].max(), zs))), 5)
        idx = temporal_sort(zs[0], 'dmean', t0=T_burn-ti)[1]
        
        for k, (z, l) in enumerate(zip(zs, test_names)):
            # Uncomment the below to sort case-by-case
            # idx = temporal_sort(z, t0=T_burn-i)[1]
            im = ax[p,k].imshow(z[:,idx].T, aspect='auto', cmap='seismic', 
                                norm=TwoSlopeNorm(0, zmin, zmax))
            cbar = fig.colorbar(im, cax=ax[p,-1])
            cbar.set_ticks([np.ceil(zmin), 0, np.floor(zmax)-1])
            ax[p,k].axvline(T_burn-ti, ls='--', c='k')
            ax[0,k].set(xticks=[], yticks=[])
            ax[0,k].set_title(l, fontsize=10)
            ax[-1,k].set(xlabel='Time (a.u.)', yticks=[])
        N = zs[0].shape[1]
        ax[p,0].set(ylabel=pop[p], yticks=[N//2, N])
    
    cmap = plt.get_cmap('plasma')
    N = tests[0][0].shape[1]
    for k, v in syl_order.items():
        for (i, t0, t1) in v:
            ax[0,k].add_patch(plt.Rectangle((t0, 0), t1-t0, -N/30, fc=cmap(i/len(v)), 
                                             clip_on=False, linewidth=0))
    fig.tight_layout()
    return fig, ax
    
def plot_ctrl_vs_nonctrl(tests, test_names, ti, tj):
    zs = [normalize(t[0][ti:tj].mean(axis=0), axis=0) for t in tests]
    fig, ax = plt.subplots(1, len(zs)-1, figsize=(1.5*(len(zs)-1), 2), 
                           sharey='all')
    for i, l in enumerate(test_names[1:]):
        ax[i].plot(zs[0], zs[i+1], 'o', ms=1)
        ax[i].plot([-2, 5], [-2, 5], c='k', ls='--')
        ax[i].set(xlabel=r'$z_{ctrl}$', xlim=[-3, 6], ylim=[-3, 6])
        ax[i].set_title(l, fontsize=10)
    ax[0].set(ylabel='z', yticks=[0, 5])
    fig.tight_layout()
    return fig, ax
    
def plot_corr_ctrl_nonctrl(tests, test_names, ti, tj, T_burn):
    # zs = [normalize(t[0][ti:tj], axis=0) for t in tests]
    zs = [t[0][ti:tj] for t in tests]
    fig, ax = plt.subplots(1, len(zs)-1, figsize=(1.5*(len(tests)-1), 2), 
                           sharey='all')
    for i, l in enumerate(test_names[1:]):
        corr = correlation(zs[0], zs[i+1], dim=1)
        ax[i].plot(corr)
        ax[i].hlines(corr[:T_burn-ti].mean(), 0, T_burn-ti, color='r')
        ax[i].hlines(corr[T_burn-ti:].mean(), T_burn-ti, tj-ti, color='r')
        ax[i].set_title(l, fontsize=10)
        ax[i].set(xlabel='Time (a.u.)')
    ax[0].set(ylabel='Corr with Ctrl')
    fig.tight_layout()
    return fig, ax
    
def plot_rate_and_change_dists(tests, test_names, ti, tj):
    ls = [t[0][ti:tj].mean(axis=0) for t in tests]
    changes = [l - ls[0] for l in ls]
    lmax = max(list(map(lambda _: _.max(), ls)))
    cmax = max(list(map(lambda _: _.max(), changes)))
    cmin = min(list(map(lambda _: _.min(), changes)))
    fig, ax = plt.subplots(len(ls), 2, figsize=(3, 1.2*len(tests)), 
                           sharex='col')
    for i, (l, c, k) in enumerate(zip(ls, changes, test_names)):
        ax[i,0].hist(l, bins=15, range=(0, lmax), density=True, log=True)
        ax[i,0].set_ylabel(k, fontsize=10)
        ax[i,1].hist(c, bins=15, range=(cmin, cmax), density=True, log=True)
    ax[-1,0].set(xlabel='Response (Hz)')
    ax[-1,1].set(xlabel='Change (Hz)')
    fig.tight_layout()
    return fig, ax