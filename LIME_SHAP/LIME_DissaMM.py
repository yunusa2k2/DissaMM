#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DissaMM
@author: yunusa2k2
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
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

bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class MultiModalClassifier(nn.Module):
    def __init__(self, img_features, text_features, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_features + text_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, img_feat, text_feat):
        x = torch.cat([img_feat, text_feat], dim=1)
        return self.fc(x)

input_dim_img = 2048
input_dim_text = bert_model.config.hidden_size
num_classes = len(class_labels)

model = MultiModalClassifier(input_dim_img, input_dim_text, num_classes).to(device)
state_dict = torch.load("DissaMM_mlp_damage.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

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

TEXT_INPUT = "i have lost most stuffs to this unfortunate flood"  # replace with real text

def predict(img_array):
    tensor = preprocess(Image.fromarray(img_array)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = resnet50(tensor)
        enc = tokenizer(TEXT_INPUT, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        text_feat = bert_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = model(img_feat, text_feat)
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