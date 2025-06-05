from data_process import *

train_loader, test_loader = load_mnist()

from model import *

device = 'cuda'
model = SimpleUnet().to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

T = 500
start = 0.0001
end = 0.05
device = 'cuda'
IMG_SIZE = 32
BATCH_SIZE = 512

betas = torch.linspace(start, end, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis = 0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 



def show_images(img_name, dataset, num_samples=10, cols=5, save = True):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(dataset):
        img = img.detach().to('cpu')
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        
        #for mnist image only, if u need colored stuff, stick with 'convert_to_image' function in source code 
        plt.imshow(img.squeeze(dim=0), cmap='gray')
    
    if (save == True):
        name = 'output' + str(img_name) + '.png'
        plt.savefig('/home/longvv/generative_model/diffusion-real/result/' + name)
    plt.show()

@torch.no_grad()
def sample_plot_image(img_name):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    img_set = []
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        # img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            img_set.append(img.squeeze(dim = 0))
            
    show_images(img_name, img_set, save = True)

model_name = 'diffusion_mnist_fixed.pth'
model_path = '/home/longvv/generative_model/models/' + model_name

from torch.optim import AdamW
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = AdamW(model.parameters(), lr=0.0005)
epochs = 60 # Try more!



import wandb
wandb.init(
    project="mnist_diffusion",       # choose any project name you like
    name="unet_diffusion_run",       # optional: gives this run a human‐friendly name
    config={
        "batch_size": BATCH_SIZE,
        "lr": 5e-4,
        "timesteps": T,
        "img_size": IMG_SIZE,
        "epochs": epochs
    }
)
config = wandb.config  # so you can refer to config.batch_size, config.lr, etc.

best_loss = 100.0
for epoch in range(epochs):

    model.train()
    for step, (X , y) in enumerate(train_loader):
        
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        t = torch.randint(0, T, (X.shape[0],), device=device).long()
        loss = get_loss(model, X, t)
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
                "train/step": step
            })
        
    if ((epoch + 1) % 5 == 0):
        sample_plot_image((epoch + 1)//5)
    
    
    
    model.eval()
    total_loss = 0
    for step, (X, y) in enumerate(test_loader):
        t = torch.randint(0, T, (X.shape[0],), device=device).long()
        loss = get_loss(model, X, t)
        total_loss += loss.item()
    
    avg_test_loss = total_loss/len(test_loader)
    print(f"Test Loss : {avg_test_loss}")
    wandb.log({ "test/loss": avg_test_loss, "test/epoch": epoch })
    print('-' * 30)

    if (avg_test_loss < best_loss):
        best_loss = avg_test_loss
        print('New best model saved !!!')
        torch.save(model.state_dict(), model_path)
