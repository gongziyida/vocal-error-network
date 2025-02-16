import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from utils import *
from matplotlib.colors import TwoSlopeNorm, Normalize

def plot_train_stats(Ws, rE, mean_HVC_input, save_W_ts, rI=None):
    ''' Plot some training stats
    '''
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
    ''' Plot some convergence stats
    '''
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

    fig, ax = plt.subplots(1, len(rEs)+1, figsize=figsize, 
                           width_ratios=([1]*len(rEs))+[0.1]) # last for cbar
    for i in range(len(rEs)):
        idx = idx_s if sort_by=='p' else idx_ds[i]
        im = ax[i].imshow(zs[i][:,idx].T, aspect='auto', cmap='hot', rasterized=True,
                          interpolation='antialiased', norm=norm)
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

def plot_ctrl_vs_nonctrl(tests, test_names, ti, tj):
    ''' scatter plots showing the joint distributions of ctrl vs nonctrl
    tests: list containing the excitatory rates
    '''
    zs = [normalize(t[ti:tj].mean(axis=0), axis=0) for t in tests]
    fig, ax = plt.subplots(1, len(zs)-1, figsize=(1.4*(len(zs)-1), 2), 
                           sharey='all')
    for i, l in enumerate(test_names[1:]):
        ax[i].plot(zs[0], zs[i+1], 'o', ms=1, color='k')
        ax[i].plot([-2, 5], [-2, 5], c='r', ls='--')
        ax[i].set(xlabel=r'$z_{Correct}$', xlim=[-3, 6], ylim=[-3, 6])
        ax[i].set_title(l, fontsize=10)
        ax[i].axes.set_aspect('equal')
    ax[0].set(ylabel='z', yticks=[0, 5])
    fig.tight_layout()
    return fig, ax
    
def plot_corr_ctrl_nonctrl(tests, test_names, ti, tj, T_burn):
    ''' Correlations between time-avg ctrl and nonctrl over time
    tests: list containing the excitatory rates
    '''
    # zs = [normalize(t[0][ti:tj], axis=0) for t in tests]
    zs = [t[ti:tj] for t in tests]
    fig, ax = plt.subplots(1, len(zs)-1, figsize=(1.5*(len(tests)-1), 2), 
                           sharey='all')
    for i, l in enumerate(test_names[1:]):
        corr = correlation(zs[0], zs[i+1], dim=1)
        ax[i].plot(corr)
        ax[i].hlines(corr[:T_burn-ti].mean(), 0, T_burn-ti, color='r')
        ax[i].hlines(corr[T_burn-ti:].mean(), T_burn-ti, tj-ti, color='r')
        ax[i].set_title(l, fontsize=10)
        ax[i].set(xlabel='Time (a.u.)')
    ax[0].set(ylabel='Corr. with \nSinging (Correct)')
    fig.tight_layout()
    return fig, ax
    
def plot_rate_and_change_dists(rEs, test_names, rE_ctrl, ti, tj):
    ''' Histograms of rates and rate changes
    '''
    ls = [rE[ti:tj].mean(axis=0) for rE in rEs]
    rE_ctrl_mean = rE_ctrl[ti:tj].mean(axis=0)
    changes = [l - rE_ctrl_mean for l in ls]
    lmax = max(list(map(lambda _: _.max(), ls)))
    cmax = max(list(map(lambda _: _.max(), changes)))
    cmin = min(list(map(lambda _: _.min(), changes)))
    fig, ax = plt.subplots(len(ls), 2, figsize=(3, 1.2*len(ls)), 
                           sharex='col')
    if len(ls) == 1:
        ax = ax[None,:]
    for i, (l, c, k) in enumerate(zip(ls, changes, test_names)):
        ax[i,0].hist(l, bins=15, range=(0, lmax), density=True, log=True)
        ax[i,0].set_ylabel(k, fontsize=10)
        ax[i,1].hist(c, bins=15, range=(cmin, cmax), density=True, log=True)
        for j in (0, 1):
            ax[i,j].set(yticks=[0.1, 1])
    ax[-1,0].set(xlabel='Response (Hz)')
    ax[-1,1].set(xlabel='Change (Hz)')
    fig.tight_layout()
    return fig, ax


def plot_raster(model1, model2, mname1, mname2, NE, cond_names, 
                t0, t1, t_on, t0_pert, t1_pert):
    ''' Plot neuronal responses for different conditions
    model1, model2: lists of neuron responses over time
    mname1, mname2: model names
    is_ff: 2-tuple specifying if model1 or 2 is EI network or not
    t0, t1: time window to plot
    t_on: song onset. Must be > 0 for sorting to work
    t0_pert, t1_pert: time window of perturbated syl
    '''
    # width and height ratios
    hr = [[1, 0.5] if md[0].shape[1]>NE else [1] for md in (model1, model2)]
    hr = hr[0] + [0.1] + hr[1] + [0.5]
    wr = [1]*len(model1) + [0.1]
    i_null = 1 + int(model1[0].shape[1]>NE)

    # preprocess data; zs has rows and cols corresponding to the img layout
    zs = []
    for md in (model1, model2):
        mean = [m.mean(axis=0)[None,:] for m in md]
        std = [m.std(axis=0)[None,:] for m in md]
        zs.append([(m[t0:t1,:NE]-a[:,:NE])/s[:,:NE] for m, a, s in zip(md, mean, std)])
        if md[0].shape[1]>NE: # Inh. as well
            zs.append([(m[t0:t1,NE:]-a[:,NE:])/s[:,NE:] for m, a, s in zip(md, mean, std)])
        zs.append(None) # For white space row
    zmin, zmax = 1e10, -1e10
    for zp in zs:
        if zp is None:
            continue
        zmin_, zmax_ = min(list(map(lambda x: x.min(), zp))), max(list(map(lambda x: x.max(), zp)))
        zmin = zmin_ if zmin > zmin_ else zmin # update
        zmax = zmax_ if zmax < zmax_ else zmax
    zmin, zmax = max(zmin, -3), min(zmax, 5)
    norm = TwoSlopeNorm(0, zmin, zmax)
    
    fig, ax = plt.subplots(len(hr), len(wr), figsize=(6, 4), width_ratios=wr, height_ratios=hr)
    for i in range(ax.shape[1]):
        ax[i_null,i].set_axis_off()
    ax[-1,-1].set_axis_off()

    #### Plotting ####
    ls = [] # for legend in the last row
    for i, zp in enumerate(zs): # row
        if zp is None:
            continue
        p = 'Exc.' if hr[i] == 1 else 'Inh.'

        # sort by the first one
        idx = temporal_sort(zp[0], 'dmean', t0=t_on)[1]

        for j, (z, l) in enumerate(zip(zp, cond_names)): 
            # plot heatmap
            im = ax[i,j].imshow(z[:,idx].T, aspect='auto', cmap='seismic', 
                                interpolation='none', norm=norm, rasterized=True)
            ax[i,j].axvline(t_on, ls='--', c='k', lw=2)
            ax[i,j].set(xticks=[], yticks=[])
            ax[0,j].set_title(l, fontsize=10, va='bottom')

            if p == 'Exc.': # plot % active in the last row
                c, label = ('C0',mname1) if i<i_null else ('C1',mname2)
                peaks = (z > 1).mean(axis=1) * 100
                l, = ax[-1,j].plot(peaks, color=c, label=label)
                if j == 0: # for legend
                    ls.append(l)
                ax[-1,j].axvline(t_on, ls='--', c='k', lw=2)
                ax[-1,j].set(xlim=[0,len(peaks)], yticks=[], xlabel='Time (ms)', 
                             xticks=[t_on, 800], xticklabels=[0, 800-t_on])
                
        if i != 0:
            ax[i,-1].set_axis_off()
        ax[i,0].set(ylabel='\n'+p, yticks=[])

    #### Color bar and labels ####
    cbar = fig.colorbar(im, cax=ax[0,-1])
    # cbar.set_ticks([np.ceil(zmin), 0, np.floor(zmax)-1])
    cbar.set_ticks([np.ceil(zmin), 0, np.floor(zmax)])
    fig.text(0.025, 0.8, mname1, rotation=90, ha='center', va='center')
    fig.text(0.025, 0.45, mname2, rotation=90, ha='center', va='center')
    # last row
    fig.legend(handles=ls, loc=(0.5, 0.22), ncols=2)
    ax[-1,0].set(yticks=[0, 40], ylabel='% excited', title='\n')
    
    #### Plot perturbed syl indicator ####
    for i in (0, -1):
        _ = max(ax[i,1].get_ylim())
        y0, height = _ * (-0.05 if i==0 else 1.05) , _ / 20 * (1 - i)
        ax[i,1].add_patch(plt.Rectangle((t0_pert, y0), t1_pert-t0_pert, height, fc='k', 
                                         clip_on=False, linewidth=0))
    return fig, ax


def plot_mean_std(ax, mean, std, a_fill, c, ls='-', xs=None, label=''):
    if xs is None:
        xs = np.arange(len(mean))
    ax.fill_between(xs, mean+std, mean-std, color=c, alpha=a_fill)
    return ax.plot(xs, mean, c=c, label=label, ls=ls)


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