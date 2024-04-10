import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import os
import sys
from tqdm import tqdm

sys.path.insert(1, './Utils/')
import Diffusion
import loader
import Unet
import cfg

@torch.no_grad()
def sample_one_model(model, config, sd, n_images=100, file_save='./',
                      times_save=[], index_run=0, dim=4):
    # Generate n_images starting points from N(0, 1)
    if dim == 4: # Assumes [B, C, H, W] for 2d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], 
                             config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3: # Assumes [B, C, N] for 1d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], 
                             config.IMG_SHAPE[1]).to(config.DEVICE)
    x_a = x_init.clone()
    
    model.eval()
    
    # Save image at 0
    path = file_save + '1000'
    # Create dir if does not exist
    doesExist = os.path.exists(path)
    if not doesExist:
        os.makedirs(path)
    torch.save(x_a, path + '/samples_a_{:d}'.format(index_run))
    
    pbar = tqdm(reversed(range(0, config.TIMESTEPS)))
    for t in pbar:
        ts = torch.ones(n_images, dtype=torch.long, device=config.DEVICE) * t
        z = torch.randn_like(x_a) if t > 1 else torch.zeros_like(x_a)
        
        # Evaluate the two models
        noise_a = model(x_a, ts)
        
        # Save image if in time to save (before update)
        if t in times_save:
            path = file_save + '{:d}'.format(t)
            # Create dir if does not exist
            doesExist = os.path.exists(path)
            if not doesExist:
                os.makedirs(path)
            torch.save(x_a, path + '/samples_a_{:d}'.format(index_run))
            torch.save(noise_a, path + '/noise_a_{:d}'.format(index_run))
        
        # Get scalings
        beta_t                            = Diffusion.get(sd.beta, ts, dim)
        one_by_sqrt_alpha_t               = Diffusion.get(sd.one_by_sqrt_alpha, ts, dim)
        sqrt_one_minus_alpha_cumulative_t = Diffusion.get(sd.sqrt_one_minus_alpha_cumulative, ts, dim) 
 
        # Diffusion with the two models but same noise
        x_a = (one_by_sqrt_alpha_t 
             * (x_a - (beta_t / sqrt_one_minus_alpha_cumulative_t) 
                * noise_a) + torch.sqrt(beta_t) * z)
    
    # Save the generated image
    path = file_save + 'generated'
    # Create dir if does not exist
    doesExist = os.path.exists(path)
    if not doesExist:
        os.makedirs(path)
    torch.save(x_a, path + '/samples_a_{:d}'.format(index_run))
    
    return x_a

# In[] Load config and data

# Config training
DATASET = 'Imagenet32'
config = cfg.load_config(DATASET)
config.DEVICE = 'cuda:2'

loading_func = 'loader.load_{:s}(config)'.format(config.DATASET)
testset = None
trainset, testset = eval(loading_func)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                               shuffle=True, num_workers=1,
                                               pin_memory=False)

dataiter = iter(trainloader)
images, labels = next(dataiter)

N = np.prod(config.IMG_SHAPE)
X = images.reshape(-1, N).float()
X = X.to(config.DEVICE)

sd = Diffusion.DiffusionConfig(
    n_steps                 = config.TIMESTEPS,
    img_shape               = config.IMG_SHAPE,
    device                  = config.DEVICE,
)


# In[] Compute the collapse time numerically

n_images = config.n_images
type_model = '{:s}_{:d}'.format(DATASET, n_images)
path_model_diffusion = '../Saves/Models/' + type_model + '/Model_350000'

model_diffusion = Unet.UNet(
    input_channels          = config.IMG_SHAPE[0],
    output_channels         = config.IMG_SHAPE[0],
    base_channels           = 64,
    base_channels_multiples = (1, 2, 4, 4),
    apply_attention         = (False, True, True, False),
    dropout_rate            = 0.1,
)
if DATASET == 'LSUN':
    model_diffusion = nn.DataParallel(model_diffusion, device_ids = [int(config.DEVICE[5])])
model_diffusion.to(config.DEVICE)
    
checkpoint = torch.load(path_model_diffusion, map_location='cpu')
model_diffusion.load_state_dict(checkpoint)
model_diffusion.to(config.DEVICE)

# Compute k nearest neighbours and collapse time
times = np.array(list(range(config.TIMESTEPS, -1, -50)))[1::]

GENERATE = True
Nsamples = 40
size = 100
 
if GENERATE:
    for i in tqdm(range(0, Nsamples)):
        path = '../Saves/Exp_collapse/' + type_model +'/'
        doesExist = os.path.exists(path)
        if not doesExist:
            os.makedirs(path)
        
        samples_a = sample_one_model(model_diffusion, config, sd, 
                                                n_images=size, 
                                                file_save=path,
                                                times_save=times,
                                                index_run=i)
else:
    # Load generated images all along the trajectories
    images_ta = torch.zeros(Nsamples*size, len(times), config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1], config.IMG_SHAPE[2])
    for i in range(0, Nsamples):
        pbar = tqdm(times)
        for (it, t) in enumerate(pbar):
            path = '../Saves/Exp_collapse/' + type_model + '/' + '{:d}/'.format(t) 
            file_a = path + '/samples_a_{:d}'.format(i)
            
            i1 = i*size
            i2 = (i+1)*size
            images_ta[i1:i2, it, :, :, :] = torch.load(file_a)
    
    # Compute the k-shortest distances between gen and training images
    N = np.prod(config.IMG_SHAPE)
    k = min(5, len(images))     # Number of nearest neighbours
    images = images.to(config.DEVICE)
    distances_tensor_all = torch.zeros(Nsamples*size, len(times), k)
    knn_tensor_all = torch.zeros(Nsamples*size, len(times), k)
    for (it, t) in enumerate(times):
        ts = torch.ones(size=(images.shape[0],), device=config.DEVICE,
                        dtype=torch.long) * t
        if t > 0:
            images_scaled = Diffusion.get(sd.sqrt_alpha_cumulative, t=ts, dim=4) * images     # Scaled data
        else:
            images_scaled = images
        
        X = images_scaled.reshape(-1, N).float() # Vector form
        for (i, img) in tqdm(enumerate(images_ta)): #images_ta)):
            Xs = img[it].reshape(-1, N).to(config.DEVICE)
            
            # Compute distances
            dist = torch.norm(Xs - X, dim=1, p=2)
            
            # k nearest neighbours
            knn = dist.topk(k, largest=False)
            
            distances_tensor_all[i, it, :] = knn[0].cpu()
            knn_tensor_all[i, it, :] = knn[1].cpu()
    
    # Extract collapse
    x = knn_tensor_all[:, :, 0].diff(1)
    tt = np.zeros(Nsamples*size, dtype=int)
    for (i,xx) in enumerate(x.numpy()):
        nz = xx.nonzero()[0]
        if len(nz) == 0:
            tt[i] = int(0)
        else:
            tt[i] = int(xx.nonzero()[0].max() + 1)
    tcol = times[tt].mean()
    err_tcol = times[tt].std()
    
    # Save times for plots
    np.save('../Saves/Exp_collapse/times/tc_{:s}.npy'.format(type_model),
            times[tt])
    

# In[] Compute phi_C(t)

from scipy import stats

times = np.array(list(range(config.TIMESTEPS, 0, -50)))

Nsamples = 40
size = 100

n_classes = 2

pbar = tqdm(times)

# Mean and std of the bootstrap resamples
means = np.zeros(len(times))
errors = np.zeros((len(times), 2))
std = np.zeros(len(times))
for (index_t, t) in enumerate(pbar):
    predictions_at = torch.ones(len(times), size*Nsamples)*(-1)
    predictions_bt = torch.ones(len(times), size*Nsamples)*(-1)
    
    path = '../Saves/Samples/' + type_model + '/{:d}/'.format(t)
    for i in np.arange(0, Nsamples): #np.delete(np.arange(0, Nsamples), 1):
        predictions_a = torch.zeros(size*Nsamples)
        predictions_b = torch.zeros(size*Nsamples)
        
        # Check if file exists, otherwise skip this index
        try:
            samples_a = torch.load(path + '/samples_a{:d}'.format(i),
                                    map_location=torch.device(config.DEVICE))
            samples_b = torch.load(path + '/samples_b{:d}'.format(i),
                                    map_location=torch.device(config.DEVICE))
        except:
            continue
    
        # Check nearest neighbour in the dataset
        for (index_img, img) in enumerate(samples_a):
            img_a = samples_a[index_img].reshape(-1, N)
            img_b = samples_b[index_img].reshape(-1, N)
            dist_a = torch.norm(X - img_a, dim=1, p=2)
            dist_b = torch.norm(X - img_b, dim=1, p=2)
            nn_a = dist_a.topk(1, largest=False)
            nn_b = dist_b.topk(1, largest=False)
            predictions_at[index_t, index_img+i*size] = nn_a[1].item()
            predictions_bt[index_t, index_img+i*size] = nn_b[1].item()
    
    # Remove the possible -1
    pat = predictions_at[index_t, predictions_at[index_t, :] > -1]
    pbt = predictions_bt[index_t, predictions_bt[index_t, :] > -1]
    
    # Compute percentage agreement
    cond = (pat == pbt)
    bootstrap  = stats.bootstrap((cond,), np.mean, n_resamples=20000)
    means[index_t] = np.mean(bootstrap.bootstrap_distribution)
    std[index_t] = bootstrap.standard_error
    
    errors[index_t][0] = bootstrap.confidence_interval[0]
    errors[index_t][1] = bootstrap.confidence_interval[1]

to_save = np.vstack((times, means, std, errors[:, 0], errors[:, 1]))
np.save('../Saves/Exp_collapse/phi_c/{:s}.npy'.format(type_model),
        to_save)
    

# In[] Compute the entropy criterion (Eq. 5)

times = np.array(list(range(config.TIMESTEPS, -1, -50)))[1:-1]

N = np.prod(config.IMG_SHAPE)
P = len(trainset)
P2 = int(4e5)       # Number of images used to evaluate the entropy
Dt = 1 - sd.alpha_cumulative.to('cpu').numpy()

# Initialisation
images = images.to(config.DEVICE)
logPt_tensor = torch.zeros(P2, len(times))

S0 = np.log(P)/N + 1/2*np.log(2*np.pi*np.exp(1)*Dt)     # Entropy of P Gaussians
for (it, t) in enumerate(times):
    
    ts = torch.ones(size=(images.shape[0],), device=config.DEVICE,
                    dtype=torch.long) * t
    if t > 0:
        images_scaled = Diffusion.get(sd.sqrt_alpha_cumulative, t=ts, dim=4) * images     # Scaled data
    else:
        images_scaled = images
    
    X = images_scaled.reshape(-1, N).float() # Vector form
    for (i, img) in tqdm(enumerate(range(P2))):
        # With trajectories from forward (assumes we have perfect score)
        mu = np.random.randint(0, P)
        Xs, _ = Diffusion.forward_diffusion(sd, images[mu], ts[mu], config)
        Xs = Xs.reshape(-1, N).to(config.DEVICE)
        
        # Compute distances
        dist = torch.norm(Xs - X, dim=1, p=2)
        
        # Compute entropy
        dst_sq = dist.cpu().numpy()**2
        mindist = dst_sq.min() 
        sumexp = np.exp(-(dst_sq-mindist)/2/Dt[times[it]]).sum()
        logPt_tensor[i, it] = -mindist/2/Dt[times[it]] + np.log(sumexp) - np.log(P) - N/2*np.log(2*np.pi*Dt[times[it]])
    
    print( ((logPt_tensor/N).mean(0)[it].item() + S0[times][it])/(np.log(P)/N) )
    
mean_crit = (logPt_tensor/N).mean(0).numpy() + S0[times]
std_crit = (logPt_tensor/N).std(0).numpy() / np.sqrt(P2)

entropies = np.vstack((times, mean_crit, std_crit))

np.save('../Saves/Exp_collapse/Entropies/{:s}'.format(type_model),
        entropies) 

