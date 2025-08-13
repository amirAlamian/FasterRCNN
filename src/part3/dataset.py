

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from config import AUGMENTATION_MULTIPLIER

class ResNetCharacterDataset(Dataset):
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = sample['image']
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.copy())
        return image, sample['label']

class AugmentedResNetDataset(Dataset):
    
    def __init__(self, base_dataset, multiplier=AUGMENTATION_MULTIPLIER, augment=True):
        self.base_dataset = base_dataset
        self.multiplier = multiplier if augment else 1
        self.augment = augment
        self.length = len(base_dataset) * self.multiplier

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        base_idx = idx % len(self.base_dataset)
        is_augmented_copy = (idx // len(self.base_dataset)) > 0
        
        image, label = self.base_dataset[base_idx]
        
        if self.augment and is_augmented_copy:
            image = self.apply_augmentation(image)
        
        return image, label
    
    def apply_augmentation(self, image_tensor):
        
        
        img = image_tensor.numpy().transpose(1, 2, 0).astype(np.float32)
        H, W, C = img.shape

        
        if random.random() < 0.75:
            angle = random.uniform(-12, 12)
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-8, 8)
            trans_x = random.uniform(-0.1, 0.1) * W
            trans_y = random.uniform(-0.1, 0.1) * H
            
            M = cv2.getRotationMatrix2D((W/2, H/2), angle, scale)
            M[0, 2] += trans_x
            M[1, 2] += trans_y
            
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

        
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2) 
            beta = random.uniform(-0.1, 0.1)  
            img = np.clip(alpha * img + beta, -3.0, 3.0)

        
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(0, 0.05), img.shape).astype(np.float32)
            img += noise

        
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        
        if random.random() < 0.5:
            erase_h, erase_w = int(random.uniform(0.05, 0.25) * H), int(random.uniform(0.05, 0.25) * W)
            erase_x, erase_y = random.randint(0, W - erase_w), random.randint(0, H - erase_h)
            img[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = np.random.uniform(-0.2, 0.2)

        
        img = np.clip(img, -3.0, 3.0)
        return torch.from_numpy(img.transpose(2, 0, 1))