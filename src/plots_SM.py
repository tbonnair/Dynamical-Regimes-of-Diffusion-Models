import torch
import sys
sys.path.insert(1, './Utils/')
import Unet
import Diffusion as dm
import loader
import cfg

import torchvision
import matplotlib.pyplot as plt
import numpy as np

def imshow(images, mean=.5, std=.5, pad_value=0, nrow=16):
    img = torchvision.utils.make_grid(images, nrow=nrow, pad_value=pad_value)
    # Unnormalize the image
    if images.shape[1] > 1:            # Multi channels
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
    else:
        img = img * std + mean      # Single channel
    
    # Plot it
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.axis('off')
    return fig


# In[] Load some generated images to plot

datasets = ['MNIST', 'CIFAR', 'Imagenet16', 'Imagenet32', 'LSUN', 'LSUN']
n_images = [10000, 3000, 1000, 1000, 20000, 100]

suffix = '{:s}_{:d}_newUnet/'

Nsamples = 16

# Load the dataset
DEVICE = 'cuda:5'

# Load generated images all along the trajectories
for i in range(len(datasets)-2):
    config = cfg.load_config(datasets[i])
    config.n_images = n_images[0]
    config.DEVICE = DEVICE
    loading_func = 'loader.load_{:s}(config)'.format(config.DATASET)
    trainset, testset = eval(loading_func)
    
    type_model = '{:s}_{:d}'.format(datasets[i], n_images[i]) 
    path = '../Saves/Samples/' + type_model + '_newUnet/900/' # time here does not matter
    file_a = path + '/samples_a0'
    images = torch.load(file_a, map_location=config.DEVICE)[0:Nsamples]
    
    fig = imshow(images.cpu(), config.mean, config.std, images.max())
    fig.savefig('../Saves/Plots/samples_{:s}.pdf'.format(datasets[i]), bbox_inches='tight')
    
# For LSUN, separate it into two
Nsamples = 8*3
for i in range(4, len(datasets)):
    config = cfg.load_config(datasets[i])
    config.n_images = n_images[0]
    config.DEVICE = DEVICE
    loading_func = 'loader.load_{:s}(config)'.format(config.DATASET)
    trainset, testset = eval(loading_func)
    
    type_model = '{:s}_{:d}'.format(datasets[i], n_images[i]) 
    path = '../Saves/Samples/' + type_model + '_newUnet/900/' # time here does not matter
    file_a = path + '/samples_a0'
    gen_images = torch.load(file_a, map_location=config.DEVICE)[0:Nsamples]
    
    fig = imshow(gen_images.cpu(), config.mean, config.std, gen_images.max(), 8)
    fig.savefig('../Saves/Plots/SM/samples_{:s}_{:d}.pdf'.format(datasets[i], n_images),
                bbox_inches='tight')


# In[] Collapse illustration
import matplotlib as mpl

mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False

datasets = ['LSUN', 'LSUN']
n_images = [20000, 100]

suffix = '{:s}_{:d}_newUnet/'
DEVICE = 'cuda:5'

# For LSUN, separate it into two
Nsamples = 8
n_nbr = 4               # Number of nearest neighour to display
distances_tensor_all = torch.zeros(len(datasets), Nsamples, n_nbr)
knn_tensor_all = torch.zeros(len(datasets), Nsamples, n_nbr)
for i in range(len(datasets)):
    # Load configuration
    config = cfg.load_config(datasets[i])
    config.n_images = n_images[i]
    config.DEVICE = DEVICE
    loading_func = 'loader.load_{:s}(config)'.format(config.DATASET)
    trainset, testset = eval(loading_func)
    
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=len(trainset),
                                              shuffle=False, 
                                              num_workers=1,
                                              pin_memory=False)
    N = np.prod(config.IMG_SHAPE)
    
    # Load all training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # Load generated images
    type_model = '{:s}_{:d}'.format(datasets[i], n_images[i]) 
    path = '../Saves/Samples/' + type_model + '_newUnet/900/' # time here does not matter
    file_a = path + '/samples_a0'
    gen_images = torch.load(file_a, map_location=config.DEVICE)[0:Nsamples]
    
    X = images.reshape(-1, N).to(config.DEVICE)
    for (j, Xs) in enumerate(gen_images):
        # Compute the distance
        Xs = Xs.clone().reshape(-1, N).to(config.DEVICE)
        dist = torch.norm(Xs - X, dim=1, p=2)
        knn = dist.topk(n_nbr, largest=False)
        distances_tensor_all[i, j] = knn[0].cpu()
        knn_tensor_all[i, j] = knn[1].cpu()
    
    # Plot
    mean = config.mean
    std = config.std
    nx = Nsamples
    ny = 5
    fig, ax = plt.subplots(nx, ny, figsize=(4, 8))
    for j in range(nx):
        img = gen_images[j].cpu() * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
        
        
        ax[j][0].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axis = ax[j][0]
        axis.xaxis.set_tick_params(labelbottom=False)
        axis.yaxis.set_tick_params(labelleft=False)
        axis.set_xticks([])
        axis.set_yticks([])
        
        for k in range(0, ny-1):
            mu1 = int(knn_tensor_all[i, j, k].item())
            img_mu1 = images[mu1].cpu() * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
            ax[j][k+1].set_xlabel(r'$d={:.1f}$'.format(distances_tensor_all[i, j, k]),
                                fontsize=8)
            
            ax[j][k+1].imshow(np.transpose(img_mu1.numpy(), (1, 2, 0)))
            axis = ax[j][k+1]
            axis.xaxis.set_tick_params(labelbottom=False)
            axis.yaxis.set_tick_params(labelleft=False)
            axis.set_xticks([])
            axis.set_yticks([])
            
            if j == 0:
                # txt = r'$\mu_{:d}$'.format(k+1)
                ax[j][k+1].set_title('$a^{{\mu_{:d}}}$'.format(k+1), fontsize=16)
            

    fig.subplots_adjust(wspace=0.1, hspace=0.)
    fig.savefig('../Saves/Plots/SM/Collapse_{:s}_{:d}.pdf'.format(datasets[i], n_images[i]), 
                bbox_inches='tight')
    # plt.show()


    