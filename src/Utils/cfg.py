import Diffusion

def load_config(DATASET):
    config = Diffusion.TrainingConfig()
    config.DATASET = DATASET             # Dataset name (MNIST, CIFAR, Imagenet, CelebA)
    
    # Fields common to all datasets
    config.DEVICE = 'cuda:0'
    config.LR = 1e-4
    config.N_STEPS = int(5e5)+1
    config.path_save = '../Saves/'
    
    if DATASET == 'MNIST':
        config.IMG_SHAPE = (1, 32, 32)
        config.BATCH_SIZE = 128
        config.path_data = '../data/'
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 10000
        
    elif DATASET == 'CIFAR':
        config.IMG_SHAPE = (3, 32, 32)
        config.BATCH_SIZE = 128
        config.path_data = '../data/'
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 6000
        
    elif DATASET == 'Imagenet16':
        config.IMG_SHAPE = (3, 16, 16)
        config.BATCH_SIZE = 128
        config.path_data = '../data/{:s}_train/'.format(DATASET)
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 1000
        
    elif DATASET == 'Imagenet32':
        config.IMG_SHAPE = (3, 32, 32)
        config.BATCH_SIZE = 128
        config.path_data = '../data/{:s}_train/'.format(DATASET)
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 1000
        
    elif DATASET == 'Imagenet64':
        config.IMG_SHAPE = (3, 64, 64)
        config.BATCH_SIZE = 64
        config.path_data = '../data/{:s}_train/'.format(DATASET)
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 1000
        
    elif DATASET == 'CelebA':
        config.IMG_SHAPE = (3, 64, 64)
        config.BATCH_SIZE = 64
        config.path_data = '../data/CelebA/img_align_celeba'
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 20000
        
    elif DATASET == 'LSUN':
        config.IMG_SHAPE = (3, 64, 64)
        config.BATCH_SIZE = 64
        config.path_data = '../data/lsun/'
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 500
        
    elif DATASET == 'SpeechCommands':
        config.IMG_SHAPE = (1, 8000)
        config.BATCH_SIZE = 64
        config.LR = 2e-5
        config.path_data = '../data/'
        config.CENTER = False
        config.STANDARDIZE = False
        config.n_images = 6000
        
    else:
        raise Exception('Dataset {:s} not implemented'.format(DATASET))
        
    return config