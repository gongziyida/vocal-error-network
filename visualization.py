import numpy as np
import matplotlib.pyplot as plt

def plot_wcol_corr(W_syl_corrs, figdim, figsize):
    fig, ax = plt.subplots(*figdim, sharex='all', sharey='all', figsize=figsize)
    ax = ax.flatten()
    l = min(len(W_syl_corrs), len(ax))
    idx = np.round(np.linspace(0, len(W_syl_corrs)-1, num=l, endpoint=True)).astype('int')
    for i, j in enumerate(idx):
        im = ax[i].imshow(W_syl_corrs[j], vmax=1, vmin=-1, cmap='seismic', aspect='auto')
        ax[i].set_title('Rendition %d' % j)
    
    fig.suptitle(r'Corr(col$_i$W , $\vec\xi_j$)')
    fig.tight_layout(pad=0.1)
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