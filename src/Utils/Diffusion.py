import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import Plot
import torchvision
import torchaudio

# ====================================================================
# Training Configuration Class
# ====================================================================
class TrainingConfig:
    '''
    TrainingConfig: Class containing all information on the data, device, LR,
    Number of SGD steps, paths for saving, etc.
    '''
    DATASET = ''                # Dataset name (MNIST, CIFAR, Imagenet, CelebA)
    IMG_SHAPE = (3, 32, 32)     # Fixed input image size
    BATCH_SIZE = 128            # Batch size
    DEVICE = 'cuda:0'           # Name of the device to be used
    LR = 1e-4                   # Learning rate
    N_STEPS = int(1e5)+1        # Number of SGD steps
    TIMESTEPS = 1000            # Define number of diffusion timesteps
    path_save = ''              # Path for saving plots and models
    path_data = ''              # Path for the data
    CENTER = True               # Whether the dataset should be centered
    STANDARDIZE = False         # Whether the dataset should be standardized
    n_images = 500              # Number of images per class
    NUM_WORKERS = 2             # Number of workers
    
    mean = 0                    # Mean of the dataset (to be computed)
    std = 0                     # Std of the dataset (to be computed)

def get(element: torch.Tensor, t: torch.Tensor, dim: int=2):
    '''
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    '''
    ele = element[t]
    if dim == 4:
        return ele.view(-1, 1, 1, 1)
    elif dim == 3:
        return  ele.view(-1, 1, 1)


# ====================================================================
# Diffusion class
# ====================================================================
class DiffusionConfig:
    '''
    ClassDiffusion: Class containing information related to the 
    diffusion process (number of steps, device, variance, etc.)
    '''
    def __init__(self, n_steps=1000, img_shape=(3, 32, 32), device='cpu'):
        self.n_steps = n_steps
        self.img_shape = img_shape
        self.device = device
        self.initialize()

    def initialize(self):
        self.beta  = self.linear_schedule()  # Linear or fixed
        self.alpha = 1 - self.beta

        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

        self.times = 1 - np.linspace(0, 1.0, self.n_steps + 1)

    def linear_schedule(self, b0=1e-4, bT=2e-2):
        '''
        Linear schedule from b0 to bT as used in Ho et al., 2020
        '''
        scale = 1000 / self.n_steps
        beta_start = scale * b0
        beta_end = scale * bT
        return torch.linspace(beta_start, beta_end, self.n_steps, 
                              dtype=torch.float32,
                              device=self.device)
    
    def fixed_schedule(self, b=6e-3):
        return torch.linspace(b, b, self.n_steps, 
                              dtype=torch.float32, device=self.device)


# ====================================================================
# Diffusion functions
# ==================================================================== 
def forward_diffusion(df, x0, timesteps, config):
    dim = len(x0.shape)
    # Generate noise realisation with the same size as a batch of images
    eps = torch.randn_like(x0)
    
    # Apply the forward diffusion kernel at times timesteps
    mean    = get(df.sqrt_alpha_cumulative, timesteps, dim) * x0        # Image scaled by exp(-t)
    std_dev = get(df.sqrt_one_minus_alpha_cumulative, timesteps, dim)   # Noise scaled by sqrt(1-exp(-2t))
    sample_a  = mean + std_dev * eps                                    # step t of the forward process
    
    # Return the noisy image and the noise realisation
    return sample_a, eps

@torch.no_grad()
def sample_diffusion_from_noise(model, n_images=25, config=TrainingConfig(), 
                                df=DiffusionConfig(), dim=3):
    
    # Generate n_images starting points from N(0, 1)
    if dim == 4: # Assumes [B, C, H, W] for 2d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], 
                             config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3: # Assumes [B, C, N] for 1d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], 
                             config.IMG_SHAPE[1]).to(config.DEVICE)
    x = x_init.clone()
    
    model.eval()
    for t in reversed(range(0, config.TIMESTEPS)):
        # Time tensor
        ts = torch.ones(n_images, dtype=torch.long, device=config.DEVICE) * t
        
        # Generate one realisation of the noise
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        
        # Predict the noise at times ts
        eps_ts = model(x, ts)
        
        # Get scaling quantities
        beta_t                            = get(df.beta, ts, dim)
        one_by_sqrt_alpha_t               = get(df.one_by_sqrt_alpha, ts, dim)
        sqrt_one_minus_alpha_cumulative_t = get(df.sqrt_one_minus_alpha_cumulative, ts, dim) 
        
        # Langevin sampling from Ho et al., 2020
        x = (one_by_sqrt_alpha_t 
             * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) 
                * eps_ts) + torch.sqrt(beta_t) * z)
    return x, x_init
    
 
#==========================================
# Training functions
#==========================================
def train_one_batch(X, model, optimizer, loss_fn, 
                    config=TrainingConfig(), 
                    sd=DiffusionConfig()):
    model.train()
    
    # Generate random times
    ts = torch.randint(low=1, high=config.TIMESTEPS, size=(X.shape[0],), device=config.DEVICE)
  
    # Extract noisy images from times t
    X_t, noise_t = forward_diffusion(sd, X, ts, config)
    X_t = X_t.to(config.DEVICE)
    
    # Apply the model
    x_ts = ts
    Y = model(X_t.float(), x_ts)
    
    # The loss is comparing the predicted and true noises
    loss = loss_fn(noise_t, Y)
    
    # Update parameters of the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Return the current loss and the batch of noisy images
    return loss.detach().item(), X_t


def train(model, trainloader, optimizer, config, sd, loss_fn,
          sweep=1., save_every=500, suffix=''):
    
    n_steps = 0     # Number of SGD steps
    
    while n_steps < config.N_STEPS:
        bar = trange(config.N_STEPS)
        
        for i, (X, _) in enumerate(trainloader):
            X = X.to(config.DEVICE) # Push X to device
            loss, _, = train_one_batch(X, model, optimizer, loss_fn, config, sd)
            n_steps += 1            # Update number of steps
            
            shallSave = (n_steps%save_every == 0)
            if n_steps >= config.N_STEPS:
                shallSave = 1
            
            if shallSave == 1:
                # Save model
                p = config.path_save + 'Models/' + suffix + 'Model_{:d}'.format(n_steps)
                torch.save(model.state_dict(), p)
                
                # Sample a small batch and save it to check quality visually
                if len(X.shape) == 4: # For images, assumes [B, C, H, W]
                    samples, samples_init = sample_diffusion_from_noise(model, 64, config, sd, dim=4)
                    fig = Plot.imshow(samples.cpu(), config.mean, config.std)
                    fig.savefig(config.path_save + 'Images/' + suffix + 'Sample_{:d}.pdf'.format(n_steps), bbox_inches='tight')
                    plt.close('all')
                elif len(X.shape) == 3: # For audio, assumes [B, C, N]
                    samples, samples_init = sample_diffusion_from_noise(model, 10, config, sd, dim=3)
                    for i in range(len(samples)):
                        torchaudio.save(config.path_save + 'Images/' + suffix + 'Sample_{:d}_{:d}.wav'.format(n_steps, i),
                                        samples[i].cpu()*config.std+config.mean, len(samples[i][0]))
            
            # Update the bar
            bar.set_description(f'loss: {loss:.5f}, n_steps: {n_steps:d}')
            
            # If we performed all the steps, exit
            if n_steps >= config.N_STEPS:
                break
            
    # Return nothing
    return
