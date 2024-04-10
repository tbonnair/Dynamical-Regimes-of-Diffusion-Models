import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import torch
import torchvision

plt.rcParams['figure.figsize'] = [6,6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight']= 'normal'
# plt.style.use('seaborn-whitegrid')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300       #Number of dpi of saved figures
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits']=(-6, 6)
mpl.rcParams['axes.formatter.use_mathtext']=True

#mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True


def imshow(images, mean=.5, std=.5):
    img = torchvision.utils.make_grid(images)
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
