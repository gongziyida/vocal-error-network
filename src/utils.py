import numpy as np
from scipy.sparse import random as srandom
from sklearn.decomposition import PCA

def generate_matrix(dim1, dim2, rand_gen, c=1, sparse=False, **kwargs):
    if c < 1:
        M = srandom(dim1, dim2, c, 'csc')
        M.data[:] = rand_gen(**kwargs, size=len(M.data))
        if not sparse:
            M = M.toarray()
    else:
        M = rand_gen(**kwargs, size=(dim1, dim2))
    return M

def block_sym_mat(N, K, var, cov):
    '''
    N: matrix size
    K: block size
    '''
    mat = np.zeros((N,N))
    for i in range(N//K):
        p, q = K*i, K*(i+1)
        mat[p:q,p:q] = cov
    if N % K > 0:
        mat[N-N%K:N,N-N%K:N] = cov
    mat[np.arange(N),np.arange(N)] = var
    return mat

def block_apply(vec, K, func):
    '''
    vec: Block-wise operation will be applied to the last dimension
    K: block size
    func: function to apply, must accept kwarg axis
    '''
    M, D = vec.shape[-1] // K, vec.shape[-1] % K
    ret = []
    for i in range(M):
        ret.append(func(vec[...,K*i:K*(i+1)], axis=-1))
    if D > 0:
        ret.append(func(vec[...,-D:], axis=-1))
    return np.stack(ret, axis=-1)

def normal_gen(rng, mean, std, size):
    return rng.normal(mean, std, size=size).clip(min=0)
    
def lognormal_gen(rng, mean, std, size):
    mean_norm = np.log(mean**2 / np.sqrt(mean**2 + std**2))
    std_norm = np.sqrt(np.log(1 + std**2 / mean**2))
    return rng.lognormal(mean_norm, std_norm, size=size)

def const_gen(val, size):
    return np.zeros(size) + val

def normalize(sig, axis, center=True):
    m = sig.mean(axis=axis, keepdims=True) if center else 0
    s = sig.std(axis=axis, keepdims=True)
    s[s==0] = 1e-5
    return (sig - m) / s

def correlation(sig1, sig2, dim=2, cosine=False): 
    ''' 
    sig1: (T1, T2, ..., Tk, N)
    sig2: (P, N) if dim == 2, or (T1, T2, ..., Tk, N) if dim == 1
    dim: int
        If 2, calculate corr(sig1[t], sig2[p]) and return (T1, T2, ..., Tk, P)
        If 1, calculate corr(sig1[t], sig2[t]) and return (T1, T2, ..., Tk) 
    cosine: bool
        If True, compute the cosine similarity instead
    '''
    if cosine:
        sig1 = sig1.copy() / np.sqrt((sig1**2).sum(axis=-1, keepdims=True))
        sig2 = sig2.copy() / np.sqrt((sig2**2).sum(axis=-1, keepdims=True))
    else:
        sig1 = normalize(sig1, -1)
        sig2 = normalize(sig2, -1)
    if dim == 1:
        corr = (sig1 * sig2).sum(axis=-1)
    elif dim == 2:
        corr = sig1 @ sig2.T 
    if not cosine: # calc mean
        corr /= sig1.shape[-1]
    assert np.nanmax(np.abs(corr)) < 1 + 1e-5
    return corr
    
def temporal_sort(r, by, t0=0):
    '''
    by: how to separate the positively and negatively modulated neurons
        can be `dmean` (change in the mean rates after t0)
        or `rmax` (sign of the peak firing rates after t0)
    t0: the start time of the stimulus
    '''
    # Negative or positive peaks
    if by == 'dmean':
        drmean = r[t0:].mean(axis=0) - r[:t0].mean(axis=0)
        mask_pos, mask_neg = drmean > 0, drmean < 0
        when_ri_peak = np.zeros(r.shape[1])
        when_ri_peak[mask_pos] = np.argmax(r[t0:, mask_pos], axis=0)
        when_ri_peak[mask_neg] = np.argmin(r[t0:, mask_neg], axis=0)
    elif by == 'rmax':
        when_ri_peak = np.argmax(np.abs(r[t0:]), axis=0)
        r_max = np.array([r[t0+t,i] for i, t in enumerate(when_ri_peak)])
        mask_pos, mask_neg = r_max > 0, r_max < 0
    else:
        raise NotImplementedError
    n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
    print(n_pos, n_neg)
    r_ret = np.zeros((r.shape[0], n_pos+n_neg))
    idx_pos = np.argsort(when_ri_peak[mask_pos])
    idx_neg = np.argsort(when_ri_peak[mask_neg])
    idx_all = np.hstack([np.where(mask_pos)[0][idx_pos], 
                         np.where(mask_neg)[0][idx_neg]])
    r_ret = r[:,idx_all]
    return r_ret, idx_all

def PCA_proj(data, n_components=3, normalize_data=True):
    ''' data : (trials, T, neurons) or (T, neurons)
    '''
    has_trial_dim = len(data.shape) == 3 # the first dimension is trial
    batch_axes = (0,1) if has_trial_dim else 0
    if normalize_data:
        data = normalize(data, axis=batch_axes)
    data_ = np.vstack(data) if has_trial_dim else data # for PCA
    pca = PCA(n_components=n_components).fit(data_)
    vec, val = pca.components_, pca.explained_variance_
    proj = data @ vec.T # (trials, T, n_components) or (T, n_components)
    # flip axis if necesary, for better visualization
    flip = proj.max(axis=batch_axes) < -proj.min(axis=batch_axes)
    vec[flip], proj[...,flip] = -vec[flip], -proj[...,flip]
    return vec, val, proj


def wasserstein_distance_exp_vs_model(mean, model_names):
    ''' Compute the Wasserstein distance between experimental data and model simulations
    '''
    from scipy.stats import wasserstein_distance_nd

    nonctrl_keys = [k for k in mean.keys() if k != 'ctrl']
    n_models = len(model_names)
    assert n_models == len(mean['ctrl'])
    
    exp_data = np.load('../experiment_data_new/processed.npz')
    exp_data = [np.stack((exp_data['mean_corr'], exp_data['mean_pert']), 1), 
                np.stack((exp_data['mean_predeaf'], exp_data['mean_postdeaf']), 1)]
    # rescale experimental data
    exp_data[0] /= exp_data[0][:,0].std()
    exp_data[1] /= exp_data[1][:,1].std()

    dist = dict()
    for j, k in enumerate(nonctrl_keys):
        dist[k] = []
        for i in range(n_models): # models
            model_data = np.stack((mean['ctrl'][i], mean[k][i]), 1)
            scale = mean['ctrl'][i].std()
            model_data = model_data / scale

            dist[k].append(wasserstein_distance_nd(model_data, exp_data[0 if 'pert' in k else 1]))
    return dist