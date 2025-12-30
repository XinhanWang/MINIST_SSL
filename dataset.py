import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from collections import Counter
from PIL import Image

def get_dataset(name, root='./data'):
    # We remove the initial transform for training data to keep them as PIL Images
    # This allows us to apply RandAugment in SemiSupervisedDataset
    
    # For Test set, we apply the standard normalization immediately
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if name == 'MNIST':
        train_dataset = datasets.MNIST(root, train=True, download=True, transform=None)
        test_dataset = datasets.MNIST(root, train=False, download=True, transform=test_transform)
    elif name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root, train=True, download=True, transform=None)
        test_dataset = datasets.FashionMNIST(root, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset {name}")
        
    return train_dataset, test_dataset

def split_labeled_unlabeled(dataset, n_labeled_per_class, num_classes=10):
    labels = np.array(dataset.targets)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs

class SemiSupervisedDataset(Dataset):
    def __init__(self, base_dataset, indices, mode='labeled'):
        self.base_dataset = base_dataset
        self.indices = indices
        self.mode = mode
        
        # Image size configuration
        self.size = 28 
        
        # === 1. Weak Augmentation ===
        self.weak_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomCrop(self.size, padding=4),
            transforms.RandomHorizontalFlip() if 'Fashion' in str(base_dataset) else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # === 2. Strong Augmentation ===
        self.strong_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomCrop(self.size, padding=4),
            transforms.RandomHorizontalFlip() if 'Fashion' in str(base_dataset) else transforms.Lambda(lambda x: x),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)) 
        ])

    def __getitem__(self, index):
        real_index = self.indices[index]
        img, target = self.base_dataset[real_index]
        
        # Ensure img is PIL Image for RandAugment
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        if self.mode == 'consistency' or self.mode == 'FreeMatch':
            # Return weak and strong augmented versions
            img_weak = self.weak_transform(img)
            img_strong = self.strong_transform(img)
            return img_weak, img_strong, target, real_index
        else:
            # Weak augmentation
            img = self.weak_transform(img)
            return img, target, real_index
            
    def __len__(self):
        return len(self.indices)

def get_dataloaders(dataset_name, n_labeled_per_class, batch_size, unlabeled_mu=1, root='./data', pin_memory=True):
    train_base, test_dataset = get_dataset(dataset_name, root)
    
    if n_labeled_per_class == -1: # Full supervised
        labeled_idxs = list(range(len(train_base)))
        unlabeled_idxs = []
        labeled_bs = batch_size
        unlabeled_bs = batch_size # Placeholder
    else:
        labeled_idxs, unlabeled_idxs = split_labeled_unlabeled(train_base, n_labeled_per_class)
        # Dynamic allocation: B_l + B_u = batch_size, B_u = mu * B_l
        # This ensures total batch size is constant (batch_size)
        labeled_bs = round(int(batch_size / (1 + unlabeled_mu)))
        if labeled_bs < 1: labeled_bs = 1
        unlabeled_bs = batch_size - labeled_bs
        if unlabeled_bs < 1: unlabeled_bs = 1
    
    labeled_dataset = SemiSupervisedDataset(train_base, labeled_idxs, mode='labeled')
    unlabeled_dataset = SemiSupervisedDataset(train_base, unlabeled_idxs, mode='unlabeled')
    consistency_dataset = SemiSupervisedDataset(train_base, unlabeled_idxs, mode='consistency') # For consistency reg
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=labeled_bs, shuffle=True, num_workers=32, pin_memory=pin_memory, drop_last=False)
    
    unlabeled_loader = None
    if len(unlabeled_idxs) > 0:
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=unlabeled_bs, shuffle=True, num_workers=32, pin_memory=pin_memory, drop_last=False)
        
    consistency_loader = None
    if len(unlabeled_idxs) > 0:
        consistency_loader = DataLoader(consistency_dataset, batch_size=unlabeled_bs, shuffle=True, num_workers=32, pin_memory=pin_memory, drop_last=False)

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=32, pin_memory=pin_memory, drop_last=False)
    
    return labeled_loader, unlabeled_loader, consistency_loader, test_loader, train_base
