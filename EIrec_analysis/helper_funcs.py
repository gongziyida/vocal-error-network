import sys
sys.path.append('../src')
import numpy as np
from utils import correlation

def get_J(net, JEE=None, JEI=None, JIE=None, JII=None):
    JEE = net.JEE.toarray() if JEE is None else JEE
    JEI = net.JEI.toarray() if JEI is None else JEI
    JIE = net.JIE.toarray() if JIE is None else JIE
    JII = net.JII.toarray() if JII is None else JII

    rmax_ratio = net.phiI[0] / net.phiE[0]

    return np.block([[JEE, -JEI * rmax_ratio], 
                     [JIE, -JII * rmax_ratio]])

def characterize_change(svds, l_or_r='left', first_n=100, th_outlier=2.5):
    if l_or_r == 'left':
        pwsim_sv = [svds[0][0].T @ sv[0][:,:first_n] for sv in svds]
    else:
        pwsim_sv = [svds[0][2] @ sv[2][:first_n].T for sv in svds]
    pwsim_sv = np.stack(pwsim_sv, 0) # (Epoch, N, first_n)
    change = 1 - np.abs(pwsim_sv).max(axis=1)
    
    # identify outlier modes
    z = change[-10:,1:].mean(axis=0) # exclude the first one which is always stable
    z = (z - z.mean()) / z.std()
    i_outlier = np.where(z > th_outlier)[0] + 1
    i_nonout = np.array([i for i in range(change.shape[1]) 
                         if i not in i_outlier and i > 0])

    return change, i_outlier, i_nonout

def characterize_memory(svds, song, l_or_r='left', th_quantile=1, method='max'):
    NE = song.shape[-1]
    if l_or_r == 'left':
        corr_song = [correlation(sv[0][:NE].T, song, dim=2, cosine=True) for sv in svds]
    else:
        corr_song = [correlation(sv[2][:,:NE], song, dim=2, cosine=True) for sv in svds]
    # stack to arrays with shape (Epoch, N, N_patterns), then abs max across pattern dim
    if method == 'max':
        memory_encode = np.abs(np.stack(corr_song,0)).max(axis=-1)
    elif method == 'mean':
        memory_encode = np.abs(np.stack(corr_song,0)).mean(axis=-1)
    else:
        raise NotImplementError
    # # choose the curve with larger abs corr
    # mask = (memory_encode_l[-1] > memory_encode_r[-1]).astype(int)[None,:]
    # memory_encode = memory_encode_l * mask + memory_encode_r * (1 - mask)

    # identify memory modes
    th_mem = np.quantile(memory_encode[0], th_quantile)
    i_memory = np.where(memory_encode[-5:].min(axis=0) > th_mem)[0]
    i_nonmem = np.array([i for i in range(memory_encode.shape[1]) if i not in i_memory])

    return memory_encode, i_memory, i_nonmem