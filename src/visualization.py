import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from utils import *
from matplotlib.colors import TwoSlopeNorm, Normalize


def plot_raster_cmp_syl_dsyl(rEs, test_names, syl, dsyl, t_start, t_end, 
                             sort_by='e', th=5, tpre=100, figsize=(2, 3)):
    ''' visualize correct and perturbed responses
    rEs: list of (T, NE) arrays. 
    sort_by: 'p' => pattern, or 'e' => error
    '''
    ti, tj = int(t_start - tpre), int(t_end)

    N = rEs[0].shape[1]
    zs = np.stack([_[ti:tj] for _ in rEs], axis=0)
    zmax = min(max(list(map(lambda _: _[0].max(), zs))), th)
    norm = Normalize(0, zmax)
        
    idx_s = np.argsort(syl)[::-1]
    idx_ds = [np.argsort(_)[::-1] for _ in dsyl]

    fig, ax = plt.subplots(1, len(rEs)+1, figsize=figsize, dpi=100, 
                           width_ratios=([1]*len(rEs))+[0.1]) # last for cbar
    for i in range(len(rEs)):
        idx = idx_s if sort_by=='p' else idx_ds[i]
        im = ax[i].imshow(zs[i][:,idx].T, aspect='auto', cmap='hot', rasterized=True,
                          interpolation='auto', norm=norm)
        ax[i].spines.top.set_visible(True)
        ax[i].spines.right.set_visible(True)
        ax[i].axvline(tpre, ls='--', c='cyan', lw=2)
        ax[i].set(yticks=[], xlabel='Time')
        ax[i].set_title(test_names[i], fontsize=10)
        
    ax[0].set_yticks([0, N-1], [1, N], rotation=90)
    ax[0].set_ylabel('E neuron index\n(sorted by change in input)')
    fig.colorbar(im, cax=ax[-1])
    ax[-1].set(yticks=[0, th//2, th], yticklabels=[0, th//2, '≥%d' % th])
    ax[-1].set_title('Hz', ha='left', fontsize=10)
    # fig.tight_layout(pad=0.5)
    return fig, ax

def plot_tests_mean(rEs, rIs, test_names, ti, tj, plot_inh=True):
    fig, ax = plt.subplots(1, len(rEs), sharey='row', sharex='all', 
                           figsize=(1.25*len(test_names), 2))
    for i, (rE, rI, l) in enumerate(zip(rEs, rIs, test_names)):
        ax[i].plot(rE[ti:tj].mean(axis=1)[:], label='E')
        if plot_inh:
            aux = rI[ti:tj, ..., None] # For compatibility with Wilson-Cowan model
            ax[i].plot(aux.mean(axis=1)[:], label='I')
        ax[i].set_title(l, fontsize=10)
        ax[i].set(xlabel='Time (a.u.)')
    ax[0].set(ylabel='Mean rate')
    ax[0].legend(ncols=2, fontsize='small', frameon=True, 
                 columnspacing=1.5, handlelength=1)
    fig.tight_layout()
    return fig, ax

def plot_corr_mat(Ws, ctrl, pert=None, sortby=None, vmin=-1, vmax=1, **ax0kwargs):
    ''' Plot the correlation between weights and song patterns
    Ws: matrix with shape (NE, NE or NI); can be sparse matrix
    ctrl, pert: control and perturbed patterns to compare, must have shape (*, NE)
    sortby: If None, no sorting; if `max` or `min`, sort by the max or min correlations 
            along the horizontal axis
    '''
    Ws = [Ws[0].toarray(), Ws[-1].toarray()] if issparse(Ws[0]) else [Ws[0], Ws[-1]]
    
    corrs = [correlation(w.T, ctrl, dim=2) for w in Ws]
    if pert is not None:
        corrs += [correlation(Ws[-1].T, pert, dim=2)]
        
    if sortby == 'min':
        idx = np.argsort(np.nanargmin(corrs[1], axis=1))
    elif sortby == 'max':
        idx = np.argsort(np.nanargmax(corrs[1], axis=1))
        
    fig, ax = plt.subplots(1, len(corrs), sharey='all', figsize=(6, 2))
    for i, j in enumerate(corrs):
        im = j if sortby is None else j[idx,:]
        im = ax[i].imshow(im, aspect='auto', interpolation='none', 
                          vmin=vmin, vmax=vmax, cmap='seismic')
    ax[0].set_title('Before learning, corr. with\ncorrect song')
    if pert is not None:
        ax[1].set_title('\ncorrect song')
        ax[2].set_title('\nperturbed song')
        fig.text(0.45, 0.92, 'After learning, corr. with')
    else:
        ax[1].set_title('After learning, corr. with\ncorrect song')
    ax[0].set(**ax0kwargs)
    cax = fig.colorbar(im, ax=ax, label='Correlation', ticks=[vmin, 0, vmax])
    return fig, ax

def plot_ctrl_vs_nonctrl(mean, se, test_names, model_names, figsize):
    ''' Plot the joint distributions between ctrl and noncontrol conditions
    mean, se: dict
        Dictionary (cases) of lists (models) of arrays (simulations). 
        Must contain the key 'ctrl'. 
    test_names: dict
        Keys are the keys of tests that are not 'ctrl'
    model_names: array-like
        Corresponds to the list in each test conditions
    '''
    nonctrl_keys = [k for k in mean.keys() if k != 'ctrl']
    n_models = len(model_names)
    assert n_models == len(mean['ctrl'])
    
    fig, ax = plt.subplots(len(nonctrl_keys), n_models, figsize=figsize, 
                           sharex='all', sharey='all')
    if len(nonctrl_keys) == 1:
        ax = ax[None,:]
    if n_models == 1:
        ax = ax[:,None]

    for i in range(n_models): # models
        ax[0,i].set_title(model_names[i])
        scale = max([v[i].std() for k, v in mean.items()]) / 2
        
        idx = np.random.choice(mean['ctrl'][i].shape[0], size=300, replace=False)
        for j, k in enumerate(nonctrl_keys):
            ax[j,0].set_ylabel(test_names[k])
            
            ax[j,i].errorbar(mean['ctrl'][i][idx]/scale, mean[k][i][idx]/scale, 
                             xerr=se['ctrl'][i][idx]/scale, yerr=se[k][i][idx]/scale, 
                             fmt='o', ms=2, c='w', mec='k', mew=1, 
                             ecolor='k', elinewidth=1, zorder=-1)

            ax[j,i].plot([-3, 50], [-3,50], c='r', ls='--', zorder=-3)
            ax[j,i].set_rasterization_zorder(0)
            ax[j,i].set(aspect=1, xticks=[], ylim=[-3,7], xlim=[-3,7])
    ax[-1,0].set(xlabel=' ')
    fig.text(0.55, 0, 'Singing correct', ha='center', va='bottom')
    return fig, ax

def plot_dist_rate_diff(ctrl, pert, t0, t1, figsize):
    ''' Plot the distributions of differential population activity
    ctrl, pert: list
        A list (models) of arrays (simulations). 
    t0, t1: int
        Singing onset and offset time indices
    '''
    from scipy.stats import wilcoxon
    fig, ax = plt.subplots(1, len(ctrl), figsize=figsize, sharey='all')
    if len(ctrl) == 1:
        ax = ax[:,None]
    for i in range(len(ctrl)): 
        z_ctrl = ctrl[i] # (trials, time, neurons)
        z_ctrl = z_ctrl[:,t0:t1].mean(axis=1) - ctrl[i][:,:t0].mean(axis=1)
        z_ctrl = z_ctrl.mean(axis=0)
        z_pert = pert[i] # (trials, time, neurons)
        z_pert = z_pert[:,t0:t1].mean(axis=1) - pert[i][:,:t0].mean(axis=1)
        z_pert = z_pert.mean(axis=0)
        diff = z_pert - z_ctrl
        diff = diff
        test = wilcoxon(diff, alternative='greater')
        print(test.pvalue)
        m = np.abs(diff).max()
        ax[i].hist(diff, bins=11, range=(-m, m), density=True)
        ax[i].set(xticks=[-int(m), 0, int(m)])
    ax[0].set(xlabel=' ', ylabel='density', yscale='log')
    return fig, ax

######## Axis-level plotting ########
def plot_mean_std(ax, mean, std, a_fill, c, ls='-', lw=1.5, xs=None, label=''):
    if xs is None:
        xs = np.arange(len(mean))
    ax.fill_between(xs, mean+std, mean-std, color=c, alpha=a_fill)
    return ax.plot(xs, mean, c=c, label=label, ls=ls, lw=lw)


def draw_traj(ax, traj, cmaps, zorders, dt=5):
    T = traj.shape[-2]
    for i, (cmap, z) in enumerate(zip(cmaps, zorders)):
        cmap = plt.get_cmap(cmap)
        for t in range(0, T, dt):
            ax.plot(*traj[i,t:t+dt+1].T, c=cmap(t/T), lw=2, zorder=z)

def draw_vec(ax, origin, vec, scale, color):
    vec = scale * vec + origin
    ap = dict(color=color, width=1, headlength=5, headwidth=5, shrink=0)
    ax.annotate('', xy=vec, xytext=origin, annotation_clip=False, arrowprops=ap)