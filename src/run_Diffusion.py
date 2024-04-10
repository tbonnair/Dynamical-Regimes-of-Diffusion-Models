import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os
import numpy as np

sys.path.insert(1, './Utils/')
import Unet
import Plot
import Diffusion
import loader
import cfg

# Config training
DATASET = 'LSUN'
config = cfg.load_config(DATASET)
config.n_images = 2000
suffix = '{:s}_{:d}_newUnet/'.format(config.DATASET,
                                     config.n_images)
config.DEVICE = 'cuda:7'

# Create path to images and model save
path_images = config.path_save + 'Images/' + suffix
path_models = config.path_save + 'Models/' + suffix
os.makedirs(path_images, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

os.system('cp run_Diffusion.py {:s}.py'.format(path_models + '_run_Diffusion.py'))
os.system('cp Utils/loader.py {:s}.py'.format(path_models + '_loader.py'))
os.system('cp Utils/cfg.py {:s}.py'.format(path_models + '_cfg.py'))

loading_func = 'loader.load_{:s}(config, loadtest=True)'.format(config.DATASET)
testset = None
trainset, testset = eval(loading_func)

# In[]

if __name__ == '__main__':
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True, 
                                              num_workers=1,
                                              pin_memory=False)
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, 
                                                  batch_size=config.BATCH_SIZE,
                                                  shuffle=False, 
                                                  num_workers=2,
                                                  pin_memory=False)

# In[] Plot one random batch of training images

# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# Plot.imshow(images, config.mean, config.std)
# plt.savefig(path_images + 'Training_set.pdf', 
#             bbox_inches='tight')

# In[] Compute Lambda

if __name__ == '__main__':
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=len(trainset),
                                              shuffle=True, 
                                              num_workers=1,
                                              pin_memory=False)
    
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # Use only the first channel
    tt = images[:,0,:,:].reshape(-1, np.prod(config.IMG_SHAPE[1::]))
    cov = torch.cov(tt.T)
    Lambda = torch.lobpcg(cov)[0].item()
    print('Largest eigenvalue is {:.4f}'.format(Lambda))


# In[] Model definition

if __name__ == '__main__':
    model = Unet.UNet(
        input_channels          = config.IMG_SHAPE[0],
        output_channels         = config.IMG_SHAPE[0],
        base_channels           = 64,
        base_channels_multiples = (1, 2, 4, 4),
        apply_attention         = (False, True, True, False),
        dropout_rate            = 0.1,
    )
    # model = nn.DataParallel(model, device_ids = config.DEVICE[0, 1])
    model.to(config.DEVICE)

if __name__ == '__main__':
    n_params = sum(p.numel() for p in model.parameters())
    print('{:.2f}M'.format(n_params/1e6))

# In[] Training and saving

if __name__ == '__main__':
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    df = Diffusion.DiffusionConfig(
        n_steps                 = config.TIMESTEPS,
        img_shape               = config.IMG_SHAPE,
        device                  = config.DEVICE,
    )
    loss_fn = nn.MSELoss()
    
    sweeping = 1.0
    save_every = 5000
    Diffusion.train(model, trainloader, optimizer, config, df, 
          loss_fn, sweeping, save_every, suffix)
    
