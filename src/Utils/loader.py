import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from operator import itemgetter

# For CelebA
import os
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted


# ===================================================================
#   Helper functions for loading datasets
# ===================================================================
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

# Create ImageNet dataset
class ImageNet(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
            
        return (x, self.labels[index])

# Code inspired by https://patrykchrabaszcz.github.io/Imagenet32/
def loadImagenet(path_to_data, img_size, classes=[],
                 transform=None):
    all_labels = []
    all_images = []
    # n_classes = len(classes)
    for idx in range(1, 10):
        data_file = path_to_data + 'train_data_batch_{:d}'.format(idx)
        d = unpickle(data_file)
        x = d['data']
        lb = np.array(d['labels'])
        
        idx_to_keep = []
        idx_to_keep = np.where(np.in1d(lb, classes))[0]
        if idx_to_keep.any():
            img_size2 = img_size * img_size
            x = np.dstack((x[idx_to_keep, :img_size2], x[idx_to_keep, img_size2:2*img_size2], x[idx_to_keep, 2*img_size2:]))
            x = x.reshape((x.shape[0], img_size, img_size, 3))#.transpose(0, 3, 1, 2)
            all_images.extend(x)
            
            # Create new labels starting from 0 to n_classes
            new_labels = np.zeros(len(idx_to_keep))
            for (it, t) in enumerate(idx_to_keep):
                new_labels[it] = classes.index(lb[t])
                
            # Save the new labels
            all_labels.extend(new_labels)
        
    # ImageNet dataset object
    trainset = ImageNet(all_images, all_labels, transform=transform)
    return trainset


def subloader(torch_set, n_images, include_list=[], props=[]):
    '''
    Return a subset of n_images images from an input dataloader excluding
    the targets not in "include_list".
    '''
    indices_keep_train = []
    targets = []
    for (i,t) in enumerate(torch_set.targets):
        if t in include_list:
            prop_class = len(np.where(np.array(targets)==include_list.index(t))[0]) / n_images
            # Keep it if we have the props
            if prop_class < props[include_list.index(t)] and len(targets) < n_images:
                indices_keep_train.append(i)
                targets.append(include_list.index(t))
    torch_set.data = torch_set.data[indices_keep_train]
    torch_set.targets = targets
    return torch_set


if __name__ == '__main__':
    img_size = 32
    path_to_data = '../data/Imagenet{:d}_train/'.format(img_size)
    transform = transforms.Compose(
        [transforms.ToTensor(),
          # transforms.RandomHorizontalFlip(p=0.5),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = loadImagenet(path_to_data, img_size, classes=[62, 367],
                     transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=len(trainset), 
                                              shuffle=False)

    # # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # # show images
    import sys
    sys.path.insert(1, './Utils/')
    import Plot
    Plot.imshow(torchvision.utils.make_grid(images))
    plt.savefig('TEST.pdf')
    
    # Compute intrinsic dimension
    import skdim
    data = images.reshape(-1, img_size*img_size*3).numpy()
    twonn =  skdim.id.TwoNN().fit(data)
    print(twonn.dimension_)
    # 23.56


# Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Get image names
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)
    self.attr = np.loadtxt(root_dir+'/../list_attr_celeba.txt', skiprows=2,
                           usecols=np.arange(1, 41))

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)
      
    label = self.attr[idx, 20]  # Male or fem
    if label == -1:
        label = 0

    return img, label





# ===================================================================
#   Loading the datasets
# ===================================================================
def load_MNIST(config, include_list=[1,8], props=[1/2, 1/2], loadtest=False):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    include_list : list, optional
        List of indices for classes that should be kept. The default is [1,7].
    props : list, optional
        Proportion of each class. The default is [1/2, 1/2].
    loadtest : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    trainset : torchvision.datasets
        Subset of the training set containing config.n_images for each class.
    testset : torchvision.datasets
        Subset of the training set containing test images for each class.
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Pad(2),
         # transforms.Resize(224),
          # transforms.Normalize((0.5,), (0.5,))
         ])
    
    trainset = torchvision.datasets.MNIST(root=config.path_data, train=True,
                                            download=False, transform=transform)
    
    # Class indices
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = itemgetter(*include_list)(classes)
    trainset = subloader(trainset, config.n_images, 
                         include_list=include_list, props=props)
    
    classes = np.arange(0, 10)
    classes = itemgetter(*include_list)(classes)
    
    trainset = subloader(trainset, config.n_images, 
                         include_list=include_list, props=props)
    
    mean = 0.0
    std = 1.0
    if config.CENTER:
        s_t = transforms.Compose([transforms.Pad(2),])
        t_data = s_t(trainset.data)
         
        mean = torch.mean(t_data / 255.)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Pad(2),
             # transforms.Resize(224),
              transforms.Normalize((mean,), (1,))
             ])
        if config.STANDARDIZE:
            std = torch.std(t_data / 255.)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Pad(2),
                  # transforms.Resize(224),
                  transforms.Normalize((mean,), (std,))
                 ])
            
        # Relaod dataset
        trainset = torchvision.datasets.MNIST(root=config.path_data, train=True,
                                                download=False, transform=transform)
        trainset = subloader(trainset, config.n_images, include_list=include_list, props=props)
        
        if loadtest:
            testset = torchvision.datasets.MNIST(root=config.path_data, train=False,
                                                   download=False, transform=transform)
            testset = subloader(testset, config.n_images, include_list=include_list, props=props)
        else:
            testset = None
        
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return trainset, testset


def load_CIFAR(config, include_list=[1,7], props=[1/2, 1/2], loadtest=False):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    include_list : list, optional
        List of indices for classes that should be kept. The default is [1,7].
    props : list, optional
        Proportion of each class. The default is [1/2, 1/2].
    loadtest : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    trainset : torchvision.datasets
        Subset of the training set containing config.n_images for each class.
    testset : torchvision.datasets
        Subset of the training set containing test images for each class.
    '''
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.RandomHorizontalFlip(p=0.5),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
    # Load train and test sets
    trainset = torchvision.datasets.CIFAR10(root=config.path_data, train=True,
                                            download=False, transform=transform)
    
    # Class indices
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = itemgetter(*include_list)(classes)
    trainset = subloader(trainset, config.n_images, 
                         include_list=include_list, props=props)
    
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    if config.CENTER:
        tmploader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                                  shuffle=True, num_workers=1)
        t_data, labels = next(iter(tmploader))
        mean = torch.mean(t_data, axis=[0, 2, 3])
        if config.STANDARDIZE:
            std = torch.std(t_data, axis=[0, 2, 3])
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
              transforms.Normalize(mean, std)
             ])
            
        # Relaod dataset
        trainset = torchvision.datasets.CIFAR10(root=config.path_data, train=True,
                                                download=False, transform=transform)
        trainset = subloader(trainset, config.n_images, 
                             include_list=include_list, props=props)
        
        if loadtest:
            testset = torchvision.datasets.CIFAR10(root=config.path_data, train=False,
                                                   download=False, transform=transform)
            testset = subloader(testset, config.n_images, 
                                include_list=include_list, props=props)
        else:
            testset = None
            
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return trainset, testset


def load_Imagenet16(config, include_list=[7,367], props=[1/2, 1/2], loadtest=False):
    return load_Imagenet(config, include_list=include_list, props=props, loadtest=False)

def load_Imagenet32(config, include_list=[7,367], props=[1/2, 1/2], loadtest=False):
    return load_Imagenet(config, include_list=include_list, props=props, loadtest=False)
    
def load_Imagenet64(config, include_list=[7,367], props=[1/2, 1/2], loadtest=False):
    return load_Imagenet(config, include_list=include_list, props=props, loadtest=False)
    
def load_Imagenet(config, include_list=[7,367], props=[1/2, 1/2], loadtest=False):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    include_list : list, optional
        List of indices for classes that should be kept. The default is [1,7].
    props : list, optional
        Proportion of each class. The default is [1/2, 1/2].
    loadtest : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    trainset : torchvision.datasets
        Subset of the training set containing config.n_images for each class.
    testset : torchvision.datasets
        Subset of the training set containing test images for each class.
    '''
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.RandomHorizontalFlip(p=0.5),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
    # Class indices
    all_data = loadImagenet(config.path_data, config.IMG_SHAPE[1], classes=include_list,
                               transform=transform)
    
    indices = []
    index_trainer = 0
    for i in range(len(include_list)):
        id_c = np.where(np.array(all_data.labels) == i)[0]
        indices.append(id_c[index_trainer*config.n_images:(index_trainer+1)*config.n_images])
    idx_trainset = np.hstack(indices)
        
    # # n_max = config.n_images*len(include_list) # Total number of images in the training set
    # id_c1 = np.where(np.array(all_data.labels) == 0)[0]
    # id_c2 = np.where(np.array(all_data.labels) == 1)[0]
    # # id_c3 = np.where(np.array(all_data.labels) == 2)[0]
    # index_trainer = 0
    # idx_trainset = np.hstack((id_c1[index_trainer*config.n_images:(index_trainer+1)*config.n_images],
    #                           id_c2[index_trainer*config.n_images:(index_trainer+1)*config.n_images]))
    # # idx_trainset = np.hstack((id_c1[index_trainer*n_images:(index_trainer+1)*n_images],
    # #                           id_c2[index_trainer*n_images:(index_trainer+1)*n_images],
    # #                           id_c3[index_trainer*n_images:(index_trainer+1)*n_images]))
        
    trainset = torch.utils.data.Subset(all_data, idx_trainset)
    testset = None
    
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    if config.CENTER:
        t_data = torch.tensor(all_data.data)[idx_trainset]
        
        mean = torch.mean(t_data / 255., axis=[0, 1, 2])
        if config.STANDARDIZE:
            std = torch.std(t_data / 255., axis=[0, 1, 2])
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Resize(224),
             transforms.Normalize(mean, std),
             ])
            
        # Reload dataset
        all_data = loadImagenet(config.path_data,
                                config.IMG_SHAPE[1],
                                classes=include_list,
                                transform=transform)
        trainset = torch.utils.data.Subset(all_data, idx_trainset)
    
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return trainset, testset


def load_CelebA(config, loadtest=False):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    include_list : list, optional
        List of indices for classes that should be kept. The default is [1,7].
    props : list, optional
        Proportion of each class. The default is [1/2, 1/2].
    loadtest : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    trainset : torchvision.datasets
        Subset of the training set containing config.n_images for each class.
    testset : torchvision.datasets
        Subset of the training set containing test images for each class.
    '''
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(config.IMG_SHAPE[1]),
         transforms.CenterCrop(config.IMG_SHAPE[1]),
         ])
    
    celeba_dataset = CelebADataset(config.path_data, transform)
    
    i1 = np.where(celeba_dataset.attr[:, 20]==-1)[0]  # "Male" is col 20, "No Beard" is 24
    i2 = np.where(celeba_dataset.attr[:, 20]==1)[0]
    indices = np.hstack((i1[0:config.n_images], i2[0:config.n_images]))
    trainset = torch.utils.data.Subset(celeba_dataset, indices)
    testset = None
    
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    if config.CENTER:
        tmploader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                                  shuffle=False, num_workers=1)
        t_data, labels = next(iter(tmploader))
        
        mean = torch.mean(t_data, axis=[0, 2, 3])
        if config.STANDARDIZE:
            std = torch.std(t_data, axis=[0, 2, 3])
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(config.IMG_SHAPE[1]),
             transforms.CenterCrop(config.IMG_SHAPE[1]),
             transforms.Normalize(mean, std)
             ])
        
        # Reload data
        celeba_dataset = CelebADataset(config.path_data, transform)
        trainset = torch.utils.data.Subset(celeba_dataset, indices)
        
        if loadtest:
            indices_test = np.hstack((i1[config.n_images:config.n_images+2000], 
                                      i2[config.n_images:config.n_images+2000]))
            testset = torch.utils.data.Subset(celeba_dataset, indices_test)
        
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return trainset, testset


def load_LSUN(config, loadtest=False):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    include_list : list, optional
        List of indices for classes that should be kept. The default is [1,7].
    props : list, optional
        Proportion of each class. The default is [1/2, 1/2].
    loadtest : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    trainset : torchvision.datasets
        Subset of the training set containing config.n_images for each class.
    testset : torchvision.datasets
        Subset of the training set containing test images for each class.
    '''
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.CenterCrop(256),
         # transforms.RandomHorizontalFlip(p=0.5),
         transforms.Resize(config.IMG_SHAPE[1]),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
    n_images = config.n_images
    classes = ['church_outdoor_train', 'conference_room_train']
    n_images_classes = [126227, 229069]     # Church and conference
    all_data = torchvision.datasets.LSUN(config.path_data, classes=classes,
                                         transform=transform)
    
    idx_trainset = np.hstack((np.arange(0*n_images, (1)*n_images), 
                              np.arange(n_images_classes[0]+n_images*0,
                                        n_images_classes[0]+n_images*(1))))
    
    
    trainset = torch.utils.data.Subset(all_data, idx_trainset)
    testset = None
    
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    if config.CENTER:
        tmploader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                                  shuffle=True, num_workers=1)
        t_data, labels = next(iter(tmploader))
        
        mean = torch.mean(t_data, axis=[0, 2, 3])
        if config.STANDARDIZE:
            std = torch.std(t_data, axis=[0, 2, 3])
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.CenterCrop(256),
             # transforms.RandomHorizontalFlip(p=0.5),
             transforms.Resize(config.IMG_SHAPE[1]),
              transforms.Normalize(mean, std)
             ])
        
        # Reload data
        all_data = torchvision.datasets.LSUN(config.path_data, classes=classes,
                                             transform=transform)
        trainset = torch.utils.data.Subset(all_data, idx_trainset)
    
    if loadtest:
        testset = torch.utils.data.Subset(all_data, np.hstack((np.arange(50000, 50000+2000),
                                                    np.arange(n_images_classes[0]+50000,
                                                              n_images_classes[0]+50000+2000))))
    
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return trainset, testset
