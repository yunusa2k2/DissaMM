#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 20:24:28 2025

@author: yunusa2k2
"""

import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 3
# class_labels = {0: "No Damage", 1: "Damage"}
# class_labels = {0: "spam/unrelate", 1: "not urgent", 2: "somewhat urgent", 3: "moderately urgent", 4: "highly urgent"}
class_labels = {0: "Irrelevant", 1: "Relevant", 2: "Uncertain"}

# ResNet50 feature extractor
resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device)
resnet50.eval()

# MLP classifier
input_dim = 2048  # ResNet50 output features
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

mlp.load_state_dict(torch.load("Dissa_mlp_relevancy.pth", map_location=device))
mlp.eval()

# Combined model: image -> ResNet50 features -> MLP
class CombinedModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        feats = self.feature_extractor(x)
        out = self.classifier(feats)
        return out

model = CombinedModel(resnet50, mlp).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4874, 0.4810, 0.4684],
                         std=[0.2465, 0.2373, 0.2398])
])

img_path = "image_test/Image197.jpg"  # replace with your image
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())
    pred_label = class_labels[pred_idx]
    confidence = probs[pred_idx].item()
    print(f"Predicted: {pred_label} (confidence: {confidence:.4f})")

background = torch.randn(8, 3, 224, 224).to(device).requires_grad_()
x_req = x.clone().requires_grad_()

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(x_req, ranked_outputs=1)

if isinstance(shap_values, (tuple, list)):
    shap_values = shap_values[0]

# Sum across channels
shap_values = np.sum(shap_values, axis=1)  # (1, 224, 224)

x_nhwc = np.transpose(x.detach().cpu().numpy(), (0, 2, 3, 1))
x_nhwc = x_nhwc * np.array([0.2465, 0.2373, 0.2398]) + np.array([0.4874, 0.4810, 0.4684])
x_nhwc = np.clip(x_nhwc, 0, 1)

# Add confidence to the label
label_with_score = f"{pred_label} ({confidence:.2%})"
shap.image_plot(shap_values, x_nhwc, labels=[label_with_score])
plt.show()