#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 2025

@author: yunusa2k2
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, mark_boundaries
from sklearn.linear_model import Ridge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_labels = {0: "No Damage", 1: "Damage"}

resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device)
resnet50.eval()

input_dim = 2048  # ResNet50 feature size
num_classes = len(class_labels)

mlp = nn.Sequential(
    nn.Linear(input_dim, 1000),
    nn.Sigmoid(),
    nn.Dropout(0.3),
    nn.Linear(1000, 20),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(20, num_classes)
).to(device)

# Load trained weights
state_dict = torch.load("Dissa_mlp_damage.pth", map_location=device, weights_only=True)
mlp.load_state_dict(state_dict)
mlp.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4874, 0.4810, 0.4684],
        std=[0.2465, 0.2373, 0.2398]
    )
])

img_path = "image_test/Image28.jpg"
img = Image.open(img_path).convert("RGB")
img_np = np.array(img.resize((224, 224)))

superpixels = quickshift(img_np, kernel_size=4, max_dist=50, ratio=0.1)
num_segments = len(np.unique(superpixels))

plt.figure(figsize=(6,6))
plt.imshow(mark_boundaries(img_np, superpixels))
plt.title(f"Superpixel Boundaries ({num_segments} segments)")
plt.axis("off")
plt.show()

def perturb_image(img_np, superpixels, mask):
    perturbed = img_np.copy()
    for seg_id, keep in enumerate(mask):
        if keep == 0:
            perturbed[superpixels == seg_id] = 128
    return perturbed

def predict(img_array):
    tensor = preprocess(Image.fromarray(img_array)).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = resnet50(tensor)
        logits = mlp(feats)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.cpu().numpy().flatten()


num_perturbations = 4
perturbation_masks_samples = np.random.binomial(1, 0.5, size=(num_perturbations, num_segments))
perturbed_images_samples = np.array([perturb_image(img_np, superpixels, mask) 
                                     for mask in perturbation_masks_samples])

predictions_samples = np.array([predict(im) for im in perturbed_images_samples])
top_classes_samples = predictions_samples.argmax(axis=1)

plt.figure(figsize=(15,5))
for i in range(num_perturbations):
    plt.subplot(1, num_perturbations, i+1)
    plt.imshow(perturbed_images_samples[i])
    plt.axis("off")
    plt.title(f"{class_labels[top_classes_samples[i]]}\nProb {predictions_samples[i, top_classes_samples[i]]:.2f}")
plt.suptitle("Sample Perturbed Images with Predicted Probabilities", fontsize=16)
plt.show()

num_perturbations_lime = 500
perturbation_masks = np.random.binomial(1, 0.5, size=(num_perturbations_lime, num_segments))
perturbed_images = np.array([perturb_image(img_np, superpixels, mask) 
                             for mask in perturbation_masks])

predictions = np.array([predict(im) for im in perturbed_images])
target_class = predictions.mean(axis=0).argmax()
class_probs = predictions[:, target_class]

ridge = Ridge(alpha=1.0)
ridge.fit(perturbation_masks, class_probs)
feature_importance = ridge.coef_

num_superpixels_to_show = 5
top_superpixels = np.argsort(feature_importance)[-num_superpixels_to_show:]
explanation_mask = np.zeros(num_segments)
np.put(explanation_mask, top_superpixels, 1)
explained_image = perturb_image(img_np, superpixels, explanation_mask)

plt.figure(figsize=(6,6))
plt.imshow(mark_boundaries(explained_image, superpixels))
plt.title(f"Top {num_superpixels_to_show} superpixels for class '{class_labels[target_class]}'")
plt.axis("off")
plt.show()