import sys
sys.path.append('../src')
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from sparse_coding_model import SparseCoding, normalize01
from train_funcs import load_models, generate_HVC
from sklearn.decomposition import DictionaryLearning
from utils import *

def make_action_basis(syllable_spectrums, n_basis=20):
    n_samples, n_syl, n_freq_bins, n_time_bins = syllable_spectrums # get shapes
    
    basis = np.zeros((n_basis, n_freq_bins*n_time_bins))
    coefs = np.zeros((n_samples*n_syl, n_basis))
    dl = DictionaryLearning(n_components=n_basis, alpha=0.1, fit_algorithm='cd', 
                            positive_code=True, positive_dict=True)
    aux = np.concatenate([syllable_spectrums[:,i].reshape(n_samples, -1) 
                          for i in range(n_syl)])
    coefs = dl.fit_transform(aux)
    a_std = coefs.std(axis=0)[None,:]
    coefs/= a_std
    basis = dl.components_ * a_std.T


def RL(env, agent, optimizer, max_epochs=2500, entropy_weight=0.05, l1_weight=0.):
    n_syl = env.t0.shape[0]
    
    actor_losses, critic_losses = [], []
    total_reward, advantages, ve_rates = [], [], []
    songs, song_embs = [], []
    
    for epoch in tqdm(range(max_epochs)):
        actor_loss, critic_loss = 0, 0
        optimizer.zero_grad()
        
        action, val, log_prob, entropy = agent()
        action[action<2] = 0 # de-noise
        bos, song_emb, rE = env.step(action) # next state
        bos_ = env._perform(action)[0] # get performance
        
        if epoch == 0 or (epoch+1) % 100 == 0: # save example
            songs.append(bos_)
            song_embs.append(song_emb)
    
        # baseline subtraction; the rE here also includes the burning period from 0 to T_burn
        reward = [rE[env.t0[t]:env.t0[t]+120].mean() for t in range(n_syl)]
        reward = np.array(reward)
        total_reward.append(np.mean(reward))
        ve_rates.append(-rE.mean())
    
        advantage = torch.tensor(reward) - val
        critic_loss += (advantage**2).sum()
        actor_loss += -(log_prob * advantage.detach()).sum()
    
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        l1 = agent.l1(l1_weight)
        loss = actor_loss + critic_loss - entropy * entropy_weight + l1
        loss.backward()
        optimizer.step()
        
        advantages.append(advantage.detach().sum())

    ret = dict(actor_loss=actor_losses, critic_loss=critic_losses, 
               total_reward=total_reward, advantage=advantages, ve_rate=ve_rates, 
               songs=songs, song_embs=song_embs)
    return ret


#### Environment and Model Classes ####
class Environment:
    VOCAL_ERR_NET_MAP = {'FF': 0, 'EI-HVC2E': 1, 'EI-E2E': 2, 'EI-E2I2E': 3}
    def __init__(self, action_basis, n_time_bins, T_song, t0s, t1s,
                 dir_sensory_net, dir_vocal_error_net, vocal_error_net_type, 
                 T_burn=500, T_post=50, spec_dt=10,
                 HVC_peak_rate=150, HVC_kernel_width=20):
        # Constants
        self.T_song = T_song
        self.T_burn = T_burn
        self.T_post = T_post
        self.T = T_burn + T_song + T_post
        self.spec_dt = spec_dt
        
        # Store action basis
        # (n_basis, n_freq_bins*n_time_bins)
        self.action_basis = action_basis
        # self.action_basis = np.concatenate(action_basis, axis=0) 
        self.action_dim = self.action_basis.shape[0]
        
        # Load sensory net
        sensory_basis = torch.load(dir_sensory_net)
        assert sensory_basis.shape[0] % n_time_bins == 0
        self.n_sensory_basis = sensory_basis.shape[1]
        self.n_time_bins = n_time_bins
        self.n_freq_bins = sensory_basis.shape[0] // n_time_bins
        self.sensory = SparseCoding(n_basis=self.n_sensory_basis, 
                                    n_freq_bins=self.n_freq_bins, 
                                    n_time_bins=self.n_time_bins)
        self.sensory.basis = sensory_basis

        # Load vocal error net
        ret = load_models(dir_vocal_error_net, 'neighbor', 'EIIE', 'mature_hvc', 0)
        self.ve_net = ret[Environment.VOCAL_ERR_NET_MAP[vocal_error_net_type]]
        self.sensory_mapping = ret[-1]
        self.ve_net_in_dim, self.HVC_dim = self.ve_net.W.shape
        
        # Construct HVC firing
        burst_ts = np.linspace(T_burn, T_burn+T_song, num=self.HVC_dim, endpoint=False)
        aux = np.zeros((self.HVC_dim,1))
        self.rH = generate_HVC(T_burn+T_song+T_post, burst_ts[:,None], 
                               HVC_peak_rate+aux, HVC_kernel_width+aux)

        self.t0, self.t1 = t0s, t1s
        self._init_ve_net()

    def _init_ve_net(self):
        rng = np.random.default_rng()
        hE0 = rng.normal(loc=-10, scale=0.5, size=self.ve_net.NE)
        hI0 = rng.normal(loc=-1, scale=0.5, size=self.ve_net.NI)
        if not hasattr(self.ve_net, 'NI'):
            self.hI0 = -1
        
        rE, rI, _, hE, hI = self.ve_net.sim(hE0, hI0=hI0, rH=self.rH[:self.T_burn], 
                                            aud=np.zeros((self.T_burn, self.ve_net_in_dim)), 
                                            save_W_ts=[], T=self.T_burn, dt=1, 
                                            noise_strength=0, no_progress_bar=True)
        self.hE0, self.hI0 = hE, hI
        
    
    def _perform(self, action):
        assert action.shape[0] == len(self.t0)
        bos = action.numpy() @ self.action_basis
        bos = bos.reshape(-1, self.n_freq_bins, self.n_time_bins).clip(min=0)
        emb = np.zeros((bos.shape[0], self.sensory_mapping.shape[-1]))
        i_active = np.where(bos.mean(axis=(1,2)) > 1e-3)[0]

        if len(i_active) > 0:
            aux = self.sensory(torch.tensor(bos[i_active], dtype=torch.float32), 
                               n_iter_coef=200)
            aux = normalize(aux[...,0].numpy(), axis=1)
            emb[i_active] = aux @ self.sensory_mapping
            if np.isnan(emb).any():
                plt.imshow(bos)
        return bos, emb

    def _reward(self, song, h0s=None):
        rE, rI, _, hE, hI = self.ve_net.sim(self.hE0, hI0=self.hI0, rH=self.rH[self.T_burn:], 
                                            aud=song, save_W_ts=[], T=self.T-self.T_burn, dt=1, 
                                            noise_strength=0, no_progress_bar=True)
        
        return -rE.mean(axis=1)
        
    def step(self, action):
        song_emb = np.zeros((self.T-self.T_burn, self.ve_net_in_dim))
        bos, emb = self._perform(action)
        for i in range(len(self.t0)):
            song_emb[self.t0[i]:self.t0[i]+100] = emb[None,i,:]
        r = self._reward(song_emb)
        return bos, song_emb, r
        
class ActorCritic(nn.Module):
    def __init__(self, n_syl, n_out, learn_std=False):
        super(ActorCritic, self).__init__()
        self.critic = nn.Parameter(torch.zeros(n_syl, dtype=torch.float) - 2)
        self.actor_mean = nn.Parameter(torch.zeros(n_syl, n_out, dtype=torch.float))
        self.actor_logstd = nn.Parameter(torch.zeros(n_syl, n_out, dtype=torch.float), 
                                         requires_grad=learn_std)
    
    def forward(self):
        val = self.critic
        action_dist = Normal(self.actor_mean, torch.exp(self.actor_logstd))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(axis=1) # (n_syl,)
        entropy = action_dist.entropy().mean()
        return action, val, log_prob, entropy
    
    def l1(self, coef):
        if coef == 0:
            return 0
        return torch.abs(self.actor_mean).mean() * coef