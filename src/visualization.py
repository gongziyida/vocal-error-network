import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from utils import *
from matplotlib.colors import TwoSlopeNorm, Normalize


def plot_raster_cmp_syl_dsyl(rEs, test_names, syl, dsyl, t_start, t_end, 
                             plot_z=False, sort_by='e', th=5,
                             tpre=100, figsize=(2, 3)):
    ''' visualize correct and perturbed responses
    rEs: list of (T, NE) arrays. 
    sort_by: 'p' => pattern, or 'e' => error
    '''
    ti, tj = int(t_start - tpre), int(t_end)

    N = rEs[0].shape[1]
    if plot_z:
        zs = np.stack([normalize(_[ti:tj], axis=0) for _ in rEs], axis=0)
        zmin = max(min(list(map(lambda _: _[0].min(), zs))), -5)
        zmax = min(max(list(map(lambda _: _[0].max(), zs))), 5)
        norm = TwoSlopeNorm(0, zmin, zmax)
    else:
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
    
def plot_tests_corrs(tests, syl_tests, syl, test_names, ti, tj, tid_perturb_input,
                     syl_order=dict(), y=0.9, cosine=False):
    ''' Correlations with syls and errors over time
    tests: list containing the excitatory rates
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
        corr = correlation(test[ti:tj], syl, dim=2)
        ax[0,i].set_title(l, fontsize=10)
        ax[-1,i].set(xlabel='Time (a.u.)')
        for j in range(corr.shape[1]):
            ax[0,i].plot(corr[:,j], c=cmap(j/syl.shape[0]))
        if i in tid_perturb_input: # non-zero error; calc corr
            if syl_ is None: # deafen
                syl_ = 0
            corr = correlation(test[ti:tj], syl_ - syl, dim=2)
            for j in range(corr.shape[1]):
                ax[1,i].plot(corr[:,j], c=cmap(j/syl.shape[0]))

    for k, v in syl_order.items():
        for (i, t0, t1) in v:
            ax[0,k].plot([t0, t1], [y, y], color=cmap(i/syl.shape[0]), lw=3)
            ax[0,k].text(t0, y - 0.1, chr(65+i), color=cmap(i/syl.shape[0]), va='top')
            
    ax[0,0].set(ylabel=r'corr$(r^E, \xi)$')
    ax[1,0].set(ylabel=r'corr$(r^E, y - \xi)$')
    fig.tight_layout()
    return fig, ax
    
def plot_tests_corrs_simple(tests, syl_tests, syl, test_names, ti, tj, tid_perturb_input,
                            syl_order=dict(), y=0.9, cosine=False):
    ''' Correlations with syls and errors over time, see `plot_tests_corrs`
    '''
    cmap = plt.get_cmap('plasma')
    n_unpert = len(tests) - len(tid_perturb_input)
    fig, ax = plt.subplots(1, len(tests), sharex='all', figsize=(1.5*len(tests), 2))
    p, q = 0, 0
    for i, (test, syl_, l) in enumerate(zip(tests, syl_tests, test_names)):
        if i in tid_perturb_input: # plot corr with error
            axi, yl = p + n_unpert, 'corr. with error'
            if syl_ is None: # deafen
                syl_ = 0
            corr = correlation(test[ti:tj], syl_ - syl, dim=2)
            p += 1
        else: # plot corr with syl pattern
            axi, yl = q, 'corr. with syllable'
            corr = correlation(test[ti:tj], syl, dim=2)
            q += 1
        
        for j in range(corr.shape[1]):
            ax[axi].plot(corr[:,j], c=cmap(j/syl.shape[0]))
        ax[axi].set_title(l, fontsize=10)
        ax[axi].set(yticks=[0, 1], ylim=[-0.5, y+0.1], ylabel=yl, xlabel='Time')

        v = syl_order.get(i)
        if v is not None:
            for (j, t0, t1) in v:
                ax[axi].plot([t0, t1], [y, y], color=cmap(j/len(v)), lw=3)
                ax[axi].text(t0, y - 0.1, chr(65+j), color=cmap(j/len(v)), va='top')

    fig.tight_layout()
    return fig, ax

def plot_ctrl_vs_nonctrl(tests, test_names, model_names, t0, t1, NE, figsize):
    ''' Plot the joint distributions between ctrl and noncontrol conditions
    tests: dict
        Dictionary (test conditions) of lists (models) of arrays (simulations). 
        Must contain the key 'ctrl'. 
    test_names: dict
        Keys are the keys of tests that are not 'ctrl'
    model_names: array-like
        Corresponds to the list in each test condition.
    t0, t1: int
        Singing onset and offset time indices
    NE: int
    '''
    nonctrl_keys = [k for k in tests.keys() if k != 'ctrl']
    n_models = len(model_names)
    assert n_models == len(tests[nonctrl_keys[0]])
    
    fig, ax = plt.subplots(len(nonctrl_keys), n_models, figsize=figsize, 
                           sharex='all', sharey='all')
    if len(nonctrl_keys) == 1:
        ax = ax[None,:]
    if n_models == 1:
        ax = ax[:,None]
        
    for i in range(n_models): # models
        ax[0,i].set_title(model_names[i])
        
        baseline = tests['ctrl'][i][:,:t0].mean(axis=(0,1))
        z_ctrl = tests['ctrl'][i][:,t0:t1].mean(axis=(0,1)) - baseline
        s = z_ctrl[:NE].std()
        z_ctrl = z_ctrl / s
        for j, k in enumerate(nonctrl_keys):
            ax[j,0].set_ylabel(test_names[k])
                
            baseline = tests[k][i][:,:t0].mean(axis=(0,1))
            z_pert = tests[k][i][:,t0:t1].mean(axis=(0,1)) - baseline
            z_pert = z_pert / s
            
            ax[j,i].scatter(z_ctrl[NE:], z_pert[NE:], s=8, c='grey', zorder=-2)
            ax[j,i].scatter(z_ctrl[:NE], z_pert[:NE], s=8, c='k', zorder=-1)
            # ax[j,i].hist2d(z_ctrl[:NE], z_pert[:NE], bins=20, norm='log', 
            #                cmap='binary', range=((0,20),(0,20)));
            ax[j,i].plot([-3, 50], [-3,50], c='r', ls='--', zorder=-3)
            ax[j,i].set_rasterization_zorder(0)
            ax[j,i].set(aspect=1, ylim=[-3,10], xlim=[-3,10], xticks=[])
    ax[-1,0].set(xlabel=' ')
    fig.text(0.55, 0, 'Singing correct', ha='center', va='bottom')
    return fig, ax

def plot_dist_rate_diff(ctrl, pert, t0, t1, NE, figsize):
    ''' Plot the distributions of differential population activity
    ctrl, pert: list
        A list (models) of arrays (simulations). 
    t0, t1: int
        Singing onset and offset time indices
    NE: int
    '''
    from scipy.stats import skewtest
    fig, ax = plt.subplots(1, len(ctrl), figsize=figsize, sharey='all')
    if len(ctrl) == 1:
        ax = ax[:,None]
    for i in range(len(ctrl)): 
        b_ctrl = ctrl[i][:,:t0].mean(axis=(0,1))
        b_pert = pert[i][:,:t0].mean(axis=(0,1))
        z_ctrl = ctrl[i][:,t0:t1].mean(axis=(0,1)) - b_ctrl
        z_pert = pert[i][:,t0:t1].mean(axis=(0,1)) - b_pert
        s = z_ctrl[:NE].std()
        z_ctrl = z_ctrl / s
        z_pert = z_pert / s
        diff = z_pert - z_ctrl
        diff = diff[:NE]
        test = skewtest(diff, alternative='greater')
        pval = test.pvalue
        print(pval)
        m = max(-diff.max(), diff.max(), 1)
        ax[i].hist(diff, bins=11, range=(-m, m), density=True)
        ax[i].set(xticks=[-int(m), 0, int(m)])
        ax[i].set_title('%.2f' % test.statistic, 
                        fontweight='bold' if test.statistic>0 else 'normal')
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