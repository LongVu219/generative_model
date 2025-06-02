import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Define your preprocessing / augmentation pipeline
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),                                # convert PIL → Tensor, scales [0,255]→[0.0,1.0]
    transforms.Normalize((0.1307,), (0.3081,)),           # standardize with MNIST mean & std
])

root_path = '/home/longvv/generative_model/data'
BATCH_SIZE = 512

def load_mnist():
    # 2. Download (or load) the MNIST training + test sets
    train_dataset = datasets.MNIST(
        root=root_path,      # where to store/download
        train=True,         # get the training split
        download=False,      # download if not already on disk
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=root_path,
        train=False,        # get the test split
        download=False,
        transform=transform,
    )

    # 3. Wrap each in a DataLoader for batching & shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,      # samples per minibatch
        shuffle=True,       # randomize order each epoch
        num_workers=4,      # how many subprocesses to use for data loading
        pin_memory=True,    # (if using GPU) speed up host→device transfer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,      # no need to shuffle during eval
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader