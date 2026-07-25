from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from typing import Tuple
from config import (
    PARAMS, 
    RAW_DATA, 
    PROCESSED_DATA,
    TRAIN_DATA,
    TEST_DATA
)
import torch
import yaml
import os


# Load params from params.yaml
with open(PARAMS, "r") as f:
    params = yaml.safe_load(f)

def compute_mean_std() -> Tuple[torch.Tensor, torch.Tensor]:
    # Load dataset with only the ToTensor transform to compute mean and std
    compute_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CIFAR10(RAW_DATA, train=True, transform=compute_transform, download=True)
    loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0
    # Get batch of images
    for images, _ in loader:
        # Get batch size
        batch_samples = images.size(0)

        # Fatten image
        images = images.view(batch_samples, images.size(1), -1)

        # Compute mean for each channel and Cumulative all batches
        mean += images.mean(2).sum(0)

    mean = mean / len(loader.dataset)

    variance = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        variance += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(variance / (len(loader.dataset)*32*32))

    return mean, std


def main():
    # Create processed folder if not exist
    os.makedirs(PROCESSED_DATA, exist_ok=True)

    transform_lst = []
    if params['base']['use_augmented']:
        transform_lst.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

    # Applying z-score
    mean, std = compute_mean_std()
    transform_lst.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    compute_transform = transforms.Compose(transform_lst)

    train_dataset = CIFAR10(RAW_DATA, train=True, transform=compute_transform, download=True)
    test_dataset = CIFAR10(RAW_DATA, train=False, transform=compute_transform, download=True)

    # Save processed data
    torch.save(train_dataset, TRAIN_DATA)
    torch.save(test_dataset, TEST_DATA)
    print(f"-> Saved data to {PROCESSED_DATA.name}")
    

if __name__ == "__main__":
    main()