import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
from tqdm import tqdm

sys.path.insert(1, './Utils')
import Diffusion
import Plot

# Gaussian score for 2 Gaussian
# forward and backward function (using true score)
def forward(x0, t):
    z = torch.randn_like(x0)
    x_t = x0*np.exp(-t) + np.sqrt(1-np.exp(-2*t))*z
    return x_t, z

def backward(x_t, t, dt, mu_star, std, z=None):
    if z == None:       # If no z is given, draw it
        z = torch.randn_like(x_t)                      # N(0,1)
    f = -x_t - 2*score(x_t, t, mu_star, std)           # Drift term
    x_tm1 = x_t - dt*f + np.sqrt(2*dt)*z
    return x_tm1, z

def score(x_t, t, mu_star, std):
    ns = x_t.shape[1]
    delta_t = 1 - np.exp(-2*t)
    Gamma_t = delta_t + std**2*np.exp(-2*t)
    mu_t = mu_star * np.exp(-t)
    m = mu_t.T @ x_t
    mu_t = mu_t.repeat(ns).reshape(-1, ns)
    return torch.tanh(m/Gamma_t)*mu_t/Gamma_t - x_t/Gamma_t

def classify(x, mu_star):
    m = mu_star.T @ x
    return torch.sign(m)

# In[] Clones experiment
DEVICE = 'cpu'
torch.manual_seed(232)
np.random.seed(232)

ds = [256, 1024, 4096, 16384]
T = 10
nsteps = 1000
dt = T / nsteps
times = np.arange(0, T, dt)
nsamples = 10000
tbranches = np.arange(1, T+.5, 0.5)

for d in ds:
    mu_star = torch.ones(d, device=DEVICE)  # Center of the Gaussians are mu_star and -mu_star
    std = 1
    phi_num = np.zeros(len(tbranches))
    all_classes = np.zeros((nsamples, len(tbranches)))
    for (ib, tbranch) in enumerate(tbranches):
        x_tm1 = torch.randn(d, nsamples, device=DEVICE)     # Sample from the source distribution
        firstTime = 1                        # Boolean to know when we branch
        for t in tqdm(reversed(times), desc='t={:.2f}'.format(tbranch)):
            if t > tbranch:
                x_tm1, _ = backward(x_tm1, t, dt, mu_star, std)
            else:
                if firstTime:
                    x_tm2 = x_tm1.clone()
                    x_tm3 = x_tm1.clone()
                    firstTime = 0
                    
                x_tm2, z2 = backward(x_tm2, t, dt, mu_star, std)
                x_tm3, z3 = backward(x_tm3, t, dt, mu_star, std)
        
        classes_clones1 = classify(x_tm2, mu_star)
        classes_clones2 = classify(x_tm3, mu_star)
        phi_num[ib] = (classes_clones1 == classes_clones2).sum() / nsamples
        all_classes[:, ib] = np.array((classes_clones1 == classes_clones2).cpu())
        
        print('Prob = {:.3f}'.format(phi_num[ib]))
    
    np.save('../Saves/Gaussian_mixtures/Speciation/phi_num_d{:d}_n{:d}'.format(d, nsamples),
            np.vstack((tbranches, phi_num)))
    np.save('../Saves/Gaussian_mixtures/Speciation/all_classes_d{:d}_n{:d}'.format(d, nsamples),
            all_classes)
