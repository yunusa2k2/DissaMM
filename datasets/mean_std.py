#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 13:27:40 2025

@author: yunusa2k2
"""

from torchvision import transforms
from PIL import Image
import torch
import os

mean = torch.zeros(3)
std = torch.zeros(3)
nb_samples = 0

dataset_dir = "coded_images/"
for subfolder in os.listdir(dataset_dir):
    subfolder_path = os.path.join(dataset_dir, subfolder)
    if os.path.isdir(subfolder_path):
        for f in os.listdir(subfolder_path):
            if f.endswith(".jpg"):
                img_path = os.path.join(subfolder_path, f)
                img = Image.open(img_path).convert("RGB")
                img = transforms.ToTensor()(img)  # convert to [C,H,W] tensor
                mean += img.view(3, -1).mean(1)
                std += img.view(3, -1).std(1)
                nb_samples += 1

mean /= nb_samples
std /= nb_samples

print("Mean:", mean)
print("Std:", std)