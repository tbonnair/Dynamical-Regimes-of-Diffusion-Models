# Code based on M. Mezard's implementation
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl

# In[]
# Initial distribution : Gaussian mixtures
# Generate P initial configurations from the superposition of two gaussians centered at 
#\pm \vec m and covariance sigma, with equal weights
def gen_gaussian_centers(N, mutilde):
    v0 = np.random.normal(size=(N))
    v0 = mutilde*v0*np.sqrt(N)/np.sqrt(np.sum(v0**2))
    return v0

def gen_a(P, N, v0, sigma):
    # Generate P samples in N dimensions centred on v0 with std sigma
    a = np.zeros((P, N))
    b = np.zeros(N)
    label = np.zeros(P, dtype=int)
    for mu in np.arange(P):
        s = np.random.choice([1,-1],p=[1/2,1/2])
        #print('test',mu)
        if(s==1):
            label[mu]=1
            b = v0 + np.random.normal(0, 1, size=(N))*sigma
        else:
            label[mu]=-1
            b = -v0 + np.random.normal(0, 1, size=(N))*sigma
        a[mu,:]=b
    return label, a

# In[] Small test
N = 64
P = 1000
mutilde = 2
sigma = 1
v0 = gen_gaussian_centers(N, mutilde)
print(v0)
print(np.sum(v0**2),N*mutilde**2)
label, a = gen_a(P, N, v0, sigma)
print(label)
print(a)

# In[] Files for determining the theoretical time of collapse using the entropy

# Sample an x from P_t(x)
def sample_from_pt(N,P,aa,t):
    x=np.zeros(N)
    mu=np.random.randint(P)
    #print(mu,aa[mu,0:3]*np.exp(-t))
    x=aa[mu,:]*np.exp(-t)+np.sqrt((1-np.exp(-2*t)))*np.random.normal(size=(N))
    return x,mu

# Pytorch version of the sampling
def sample_from_pt_torch(N, P, aa, t, ntry=1000, device='cpu'):
    mu = np.random.randint(P, size=ntry)        # Choose ntry mu
    z = torch.randn(ntry, N, device=device)     # Random matrix of size ntry x N 
    delta_t = (1-np.exp(-2*t))                  # Variance
    x = aa[mu,:]*np.exp(-t)+np.sqrt(delta_t)*z  # Sample
    return x, mu

# Compute (1/N)*log(P_t(x))
def calc_log_pt(N,P,aa,t,x):
    aat=aa*np.exp(-t)
    dist=np.zeros(P)
    uu=np.ones(P)
    dist1=np.sum(x**2)*uu
    dist2=np.sum(aat**2,axis=1)
    dist3=aat@x
    dist=dist1+dist2-2*dist3
    distmin=np.min(dist)
    dist=dist-distmin
    lam=1/(1-np.exp(-2*t))
    sumg=np.sum(np.exp(-lam*dist/2))
    res=np.log(sumg)-lam*distmin/2-np.log(P)
    #print('test',np.log(sumg),lam*distmin/2,res,distmin/N,1-np.exp(-2*t))
    return res/N-(1/2)*np.log(2*np.pi*(1-np.exp(-2*t)))

# Compute (1/N)*log(P_t(x))  
def calc_log_pt_torch(N, P, aa, t, x, device='cpu'):
    aat = aa*np.exp(-t)
    dist = torch.zeros(P, ntry, device=device)
    uu = torch.ones(P, ntry, device=device)
    dist1 = torch.sum(x**2, 1)*uu
    dist2 = torch.sum(aat**2, axis=1)
    dist3 = aat@x.T
    dist = dist1 - 2*dist3
    dist += dist2.unsqueeze(1).repeat(1, ntry)  # Duplicate columns
    distmin, indices = torch.min(dist, 0)
    dist = dist - distmin
    lam = 1/(1-np.exp(-2*t))
    sumg = torch.sum((-lam*dist/2).exp(), 0)
    res = sumg.log() - lam*distmin/2 - np.log(P)
    f = (res/N - (1/2)*np.log(2*np.pi*(1-np.exp(-2*t))))
    return f

# In[] Theoretical versus numerical

Ns = [256, 1024, 4096]  # N>4096 fail on my GPUs (maybe reduce P or ntry)
P = 20000
mutilde = 1             # Order 1
sigma = 1               # Order 1

numt = 20               # Number of time samples for numerics
numt_theo = 200         # Number of time samples for theoretical curve
ntry = 5000             # Number of trials (control the error bar sizes)
dupl = 50               # Number of time we evaluate ntry (in the end, we have ntry*dupl evaluation to average over)
DEVICE = 'cuda:0'       # Torch device ('cpu' or 'cuda:N')
tinit = 8               # Maximum time (T)

crit_N = []
crit_theo_N = []
errors_N = []

for N in Ns:
    print('N=', N, 'P=', P, 'alpha=', np.log(P)/N, 'mutilde=', mutilde, 'sigma=', sigma)
    v0 = gen_gaussian_centers(N, mutilde)
    label,aa = gen_a(P, N, v0, sigma)
    print('Initial aa generated')
    
    alpha = np.log(P)/N
    
    # Torch input
    aa = torch.tensor(aa, device=DEVICE)
    crit = np.zeros(numt)
    error = np.zeros(numt)
    tt = np.zeros(numt)
    np.random.seed(421)
    
    for it in np.arange(numt):
        t = tinit*(1-(it/numt)*.98)
        tt[it] = t
        
        # Tensorized version
        resg = []
        for n in range(dupl):
            x, mu = sample_from_pt_torch(N, P, aa, t, ntry, device=DEVICE)
            res = calc_log_pt_torch(N, P, aa, t, x, device=DEVICE)
            resg.extend(res.cpu().numpy())
        resg = np.array(resg)
        I = resg.mean()
        Ilt = -(1/2+(1/2)*np.log(2*np.pi*(1-np.exp(-2*t))))
        crit[it] = I - Ilt + alpha
        error[it] = resg.std()/np.sqrt(ntry*dupl)
        print('time', t, 'crit', crit[it])
        
    # Theoretical expression
    crit_theo = np.zeros(numt_theo)
    tt_theo = np.zeros(numt_theo)
    
    for it in np.arange(numt_theo):
        t = tinit*(1-(it/numt_theo)*.98)
        tt_theo[it] = t
        delta = 1-np.exp(-2*t)
        sigt = sigma*np.exp(-t)
        crit_theo[it] = np.log(P)/N+(1/2)*np.log(delta/(delta+sigt**2))
        if crit_theo[it] < 0:
            crit_theo[it] = 0
    print('Computation of crit_theo done')
    
    crit_N.append(crit/alpha)
    errors_N.append(error/alpha)
    crit_theo_N.append(crit_theo/alpha)

to_save_num = np.vstack((tt, crit_N, errors_N))
to_save_th = np.vstack((tt_theo, crit_theo_N))
np.save('../Saves/Gaussian_mixtures/Collapse/f(t)_num_GM.npy', to_save_num)
np.save('../Saves/Gaussian_mixtures/Collapse/f(t)_theo.npy', to_save_th)

