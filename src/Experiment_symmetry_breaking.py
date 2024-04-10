import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
from scipy import stats
import torchvision

import sys
sys.path.insert(1, './Utils/')
import Unet
import Diffusion as dm
import cfg

# Config training
DATASET = 'Imagenet32'
config = cfg.load_config(DATASET)
suffix = '{:s}_{:d}_newUnet/'.format(config.DATASET,
                                     config.n_images)
config.DEVICE = 'cuda:0'

path_images = '../Saves/Images/'

@torch.no_grad()
def sample_branching(model_diffusion, sd, time_branch,  n_images=250, dim=4):
    
    # Generate n_images starting points from N(0, 1)
    if dim == 4: # Assumes [B, C, H, W] for 2d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], 
                             config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3: # Assumes [B, C, N] for 1d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], 
                             config.IMG_SHAPE[1]).to(config.DEVICE)
        
    x = x_init.clone()
    
    model_diffusion.eval()
    
    # First part of the trajectory T -> time_branch
    pbar = tqdm(reversed(range(time_branch, config.TIMESTEPS)))
    for t in pbar:
        pbar.set_description('Before branching')
        ts = torch.ones(n_images, dtype=torch.long, device=config.DEVICE) * t
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        
        predicted_noise = model_diffusion(x, ts)
        
        beta_t                            = dm.get(sd.beta, ts, dim)
        one_by_sqrt_alpha_t               = dm.get(sd.one_by_sqrt_alpha, ts, dim)
        sqrt_one_minus_alpha_cumulative_t = dm.get(sd.sqrt_one_minus_alpha_cumulative, ts, dim) 
 
        x = (one_by_sqrt_alpha_t 
             * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) 
                * predicted_noise) + torch.sqrt(beta_t) * z)
    
    # From there generate two trajectories with different noise
    x1 = x.clone()
    x2 = x.clone()
    pbar = tqdm(reversed(range(0, time_branch)))
    for t in pbar:
        pbar.set_description('After branching')
        ts = torch.ones(n_images, dtype=torch.long, device=config.DEVICE) * t
        
        z1 = torch.randn_like(x1) if t > 1 else torch.zeros_like(x1)
        z2 = torch.randn_like(x2) if t > 1 else torch.zeros_like(x2)
        
        noise1 = model_diffusion(x1, ts)
        noise2 = model_diffusion(x2, ts)
        
        beta_t                            = dm.get(sd.beta, ts, dim)
        one_by_sqrt_alpha_t               = dm.get(sd.one_by_sqrt_alpha, ts, dim)
        sqrt_one_minus_alpha_cumulative_t = dm.get(sd.sqrt_one_minus_alpha_cumulative, ts, dim) 
 
        x1 = (one_by_sqrt_alpha_t 
             * (x1 - (beta_t / sqrt_one_minus_alpha_cumulative_t) 
                * noise1) + torch.sqrt(beta_t) * z1)
        
        x2 = (one_by_sqrt_alpha_t 
             * (x2 - (beta_t / sqrt_one_minus_alpha_cumulative_t) 
                * noise2) + torch.sqrt(beta_t) * z2)
    return x1, x2

# In[] Load diffusion and classification models

type_model = suffix
path_model_diffusion = '../Saves/Models/' + type_model + '/Model_350000'

# Load classifier and Lambda
if DATASET == 'MNIST':
    # Lambda = 7.66
    path_model_classification = '../Saves/Models/Classification/ResNet18_MNIST18_centered'
elif DATASET == 'LSUN64':
    # Lambda = 60.5
    path_model_classification = '../Saves/Models/Classification/ResNet18_LSUN64_centred'
elif DATASET == 'CelebA64':
    # Lambda = 116.72
    path_model_classification = '../Saves/Models/Classification/ResNet18_CelebA64_centred'
elif DATASET == 'CIFAR':
    # Lambda = 16.72
    path_model_classification = '../Saves/Models/Classification/ResNet18_CIFAR32'
elif DATASET == 'Imagenet32' or DATASET == 'Imagenet16':
    # if config.IMG_SHAPE[1] == 32:
        # Lambda = 17.16
    # elif config.IMG_SHAPE[1] == 16:
        # Lambda = 3.05
    path_model_classification = '../Saves/Models/Classification/ResNet18_Imagenet{:d}_centred'.format(config.IMG_SHAPE[1])
# ==========================================

df = dm.DiffusionConfig(
    n_steps                 = config.TIMESTEPS,
    img_shape               = config.IMG_SHAPE,
    device                  = config.DEVICE,
)

model_diffusion = Unet.UNet(
    input_channels          = config.IMG_SHAPE[0],
    output_channels         = config.IMG_SHAPE[0],
    base_channels           = 64,
    base_channels_multiples = (1, 2, 4, 4),
    apply_attention         = (False, True, True, False),
    dropout_rate            = 0.1,
)
# model_diffusion = nn.DataParallel(model_diffusion, device_ids = [int(DEVICE[5])]) # If Multi-GPU
model_diffusion.to(config.DEVICE)

checkpoint = torch.load(path_model_diffusion, map_location='cpu')
model_diffusion.load_state_dict(checkpoint)
model_diffusion.to(config.DEVICE)

# Classification
if DATASET in ('MNIST'):
    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            base = torchvision.models.resnet18(pretrained=True)
            # FOR MNIST
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.base = nn.Sequential(*list(base.children())[1:-1])
            
            # self.base = nn.Sequential(*list(base.children())[0:-1])
            in_features = base.fc.in_features
            self.drop = nn.Dropout()
            self.final = nn.Linear(in_features, 2)
        
        def forward(self,x):
            x = self.conv1(x)
            x = self.base(x)
            x = self.drop(x.view(-1,self.final.in_features))
            return self.final(x)
else:
    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            base = torchvision.models.resnet18(pretrained=True)
            self.base = nn.Sequential(*list(base.children())[:-1])
            in_features = base.fc.in_features
            self.drop = nn.Dropout()
            self.final = nn.Linear(in_features, 2)
        
        def forward(self,x):
            x = self.base(x)
            x = self.drop(x.view(-1,self.final.in_features))
            return self.final(x)
    
model_classification = ResNet()
checkpoint = torch.load(path_model_classification, map_location='cpu')
model_classification.load_state_dict(checkpoint)
model_classification.to(config.DEVICE)

# In[] Branching experiment

times = np.array(list(range(config.TIMESTEPS, 0, -50)))#[1::]

GENERATE = True
Nsamples = 40
size = 100

tf = torchvision.transforms.Resize(224)
n_classes = 2

if GENERATE:
    check_done = False
    for i in range(0, Nsamples):
        pbar = tqdm(times)
        for t in pbar:
            # Create repository
            pbar.set_description('t={:d}'.format(t))
            path = '../Saves/Samples/' + type_model + '/{:d}/'.format(t)
            doesExist = os.path.exists(path)
            if not doesExist:
                os.makedirs(path)
            
            samples_a, samples_b = sample_branching(model_diffusion, df, t, 
                                                    n_images=size)
            torch.save(samples_a, path + '/samples_a{:d}'.format(i))
            torch.save(samples_b, path + '/samples_b{:d}'.format(i))
else:
    pbar = tqdm(times)
    predictions_at = torch.zeros(len(times), size*Nsamples)
    predictions_bt = torch.zeros(len(times), size*Nsamples)
    
    # Mean and std of the bootstrap resamples
    means = np.zeros(len(times))
    errors = np.zeros((len(times), 2))
    std = np.zeros(len(times))
    for (index_t, t) in enumerate(pbar):
        predictions_a = torch.zeros(size*Nsamples)
        predictions_b = torch.zeros(size*Nsamples)
        path = '../Saves/Samples/' + type_model + '/{:d}/'.format(t)
        for i in np.arange(0, Nsamples):
            samples_a = torch.load(path + '/samples_a{:d}'.format(i),
                                   map_location=torch.device(config.DEVICE))
            samples_b = torch.load(path + '/samples_b{:d}'.format(i),
                                   map_location=torch.device(config.DEVICE))
        
            # Resize images for classification
            samples_a = tf(samples_a)
            samples_b = tf(samples_b)
            
            prediction_a = model_classification(samples_a)
            prediction_a = torch.argmax(prediction_a, dim=1).cpu()
            prediction_b = model_classification(samples_b)
            prediction_b = torch.argmax(prediction_b, dim=1).cpu()
            
            predictions_at[index_t, size*i:size*(i+1)] = prediction_a
            predictions_bt[index_t, size*i:size*(i+1)] = prediction_b
        
        # Compute percentage agreement
        cond = (predictions_at[index_t, :] == predictions_bt[index_t, :])
        bootstrap  = stats.bootstrap((cond,), np.mean, n_resamples=20000)
        means[index_t] = np.mean(bootstrap.bootstrap_distribution)
        # std[index_t] = np.array(cond).astype(float).std()
        std[index_t] = bootstrap.standard_error
        
        errors[index_t][0] = bootstrap.confidence_interval[0]
        errors[index_t][1] = bootstrap.confidence_interval[1]
    
    to_save = np.vstack((means, std, errors[:, 0], errors[:, 1]))
    
    # Save for further analysis and plot
    np.save('../Saves/Exp_spec/{:s}'.format(type_model),
            to_save)


