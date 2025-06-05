from data_process import *

train_loader, test_loader = load_mnist()

from model import *

device = 'cuda'

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

from torchvision import models
model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(
    in_channels=2,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)
model.maxpool = nn.Identity()
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))


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

crit = nn.CrossEntropyLoss()
def get_loss(model, x_0, y, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    timestep = t/T
    timestep = timestep.view(-1, 1, 1, 1).expand(-1, x_0.shape[1], x_0.shape[2], x_0.shape[3])
    X = torch.cat([x_noisy, timestep], dim = 1)
    pred = model(X)
    return crit(pred, y)
    

from torch.optim import AdamW
epochs = 100
lr = 1e-4
optimizer = AdamW(model.parameters(), lr=lr)


model_name = 'diffusion_classifier.pth'
model_path = '/home/longvv/generative_model/models/' + model_name

best_loss = 100.0
for epoch in range(epochs):

    model.train()
    for step, (X , y) in enumerate(train_loader):
        
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # normalize t0 to 0....1
        t = torch.randint(0, T, (X.shape[0],), device=device).long()
        
        loss = get_loss(model, X, y, t)
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
    
    
    
    model.eval()
    total_loss = 0
    for step, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)
        t = torch.randint(0, T, (X.shape[0],), device=device).long()
        loss = get_loss(model, X, y, t)
        total_loss += loss.item()
    
    avg_test_loss = total_loss/len(test_loader)
    print(f"Test Loss : {avg_test_loss}")
    print('-' * 30)

    if (avg_test_loss < best_loss):
        best_loss = avg_test_loss
        print('New best model saved !!!')
        torch.save(model.state_dict(), model_path)

