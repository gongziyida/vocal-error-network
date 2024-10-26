import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

mse = nn.MSELoss(reduction='mean')

def normalize01(x, dims):
    xmin = x.min(dim=dims, keepdims=True).values
    xmax = x.max(dim=dims, keepdims=True).values
    x = (x - xmin) / (xmax - xmin)
    # x = x / x.std(dim=dims, keepdims=True)
    return x

class SparseCoding(nn.Module):
    def __init__(self, n_basis, n_freq_bins, n_time_bins, 
                 sparse_loss='l1', lam=0.2, device=None):
        super(SparseCoding, self).__init__()
        self.n_basis = n_basis
        self.n_freq_bins, self.n_time_bins = n_freq_bins, n_time_bins
        self.device = torch.device('cpu') if device is None else device
        self.basis = torch.randn((n_freq_bins * n_time_bins, n_basis), 
                                 requires_grad=True).to(device)

        self.basis_optimizer = torch.optim.Adam([{'params': self.basis, 'lr': 1e-3}])

        if sparse_loss == 'smooth':
            self.sparse_loss = lambda x: torch.log(1+x**2).mean()
        elif sparse_loss == 'l1':
            self.sparse_loss = lambda x: torch.abs(x).mean()
        self.lam = lam

    def _response(self, spec, n_iter):
        # assume spec is already normalized
        coef = torch.zeros((spec.shape[0], self.n_basis), 
                        requires_grad=True, device=self.device)
        with torch.no_grad(): # initialize the coefficients
            coef.data += 0.5 / self.n_basis
            # Or, solve BA = X^T for initial condition
            # coef.data = torch.linalg.lstsq(self.basis, spec.T).solution.T
        sig = spec.std()

        # Optimize the coefficients on reconstruction and sparsity, given the basis
        self.basis.requires_grad = False # need to disable when opt coef
        optimizer = torch.optim.Adam([{'params': coef, 'lr': 1e-3}])
        old_loss = 1e-10
        for _ in range(n_iter):
            optimizer.zero_grad()
            pred = coef @ self.basis.T
            loss = mse(pred, spec) + self.lam * sig * self.sparse_loss(coef / sig)
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #     coef /= coef.std(dim=1, keepdims=True)
            if np.abs(old_loss - loss.item()) / old_loss < 1e-3:
                break
            old_loss = loss.item()
        self.basis.requires_grad = True # set it back
        coef = coef.detach()
        return coef
        
    def _single_pass(self, spec, n_iter_coef, n_iter_basis):
        # assume spec originally has shape (batch_size, n_freq_bins * n_time_bins)
        spec = normalize01(spec, dims=-1) # don't move it to self.response
        coef = self._response(spec, n_iter_coef) # (batch_size, n_basis)
        # Optimize the basis on reconstruction given the coefficients
        for i in range(n_iter_basis):
            self.basis_optimizer.zero_grad()
            pred = coef @ self.basis.T
            loss = mse(pred, spec)
            loss.backward()
            self.basis_optimizer.step()
        with torch.no_grad():
            self.basis /= self.basis.std(dim=0, keepdims=True)
        return coef, loss.item()
        
    def forward(self, spec, n_iter_coef=30, pad=0, stride=1, train=False, n_iter_basis=50):
        if len(spec.shape) == 3: # (batch_size, n_freq_bins, n_time_bins) 
            batch_size = spec.shape[0]
        elif len(spec.shape) == 2: # (n_freq_bins, n_time_bins)
            batch_size = 1
            spec = spec[None,...]
        T = spec.shape[-1]
        # (n_freq_bins, batch_size, n_time_bins)
        spec_ = torch.zeros((batch_size,self.n_freq_bins,T+pad*2))
        spec_[...,pad:T+pad] = spec

        coef, dbasis = [], [] # dbasis is the change of basis
        for t in range(0, T-self.n_time_bins+1, stride):
            x = spec_[...,t:t+self.n_time_bins].flatten(-2, -1)
            if train:
                ret = self._single_pass(x, n_iter_coef, n_iter_basis)
                coef.append(ret[0])
                dbasis.append(ret[1])
            else:
                x = normalize01(x, dims=-1) # don't move it to self.response
                coef.append(self._response(x, n_iter_coef))
        
        coef = torch.stack(coef, dim=-1) # (batch_size, n_basis, time)
        if train:
            return coef, dbasis
        else:
            return coef