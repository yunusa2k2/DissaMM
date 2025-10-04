#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import shap
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class_labels = {0: "spam/unrelate", 1: "not urgent", 2: "somewhat urgent", 3: "moderately urgent", 4: "highly urgent"}
class_labels = {0: "Irrelevant", 1: "Relevant", 2: "Uncertain"}


# ResNet50 feature extractor
resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
in_features_img = resnet50.fc.in_features
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device)
resnet50.eval()

# BERT (frozen)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# Multi-modal classifier
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

num_classes = 3
classifier = MultiModalClassifier(in_features_img, bert_model.config.hidden_size, num_classes).to(device)

# Load trained weights
classifier.load_state_dict(torch.load("DissaMM_mlp_relevancy.pth", map_location=device))
classifier.eval()

class MultiModalPipelineForSHAP(nn.Module):
    def __init__(self, resnet, bert, classifier, fixed_input_ids, fixed_attention_mask):
        super().__init__()
        self.resnet = resnet
        self.bert = bert
        self.classifier = classifier
        self.fixed_input_ids = fixed_input_ids
        self.fixed_attention_mask = fixed_attention_mask

    def forward(self, images):
        img_features = self.resnet(images)
        text_features = self.bert(
            input_ids=self.fixed_input_ids,
            attention_mask=self.fixed_attention_mask
        ).pooler_output
        text_features = text_features.repeat(img_features.size(0), 1)
        return self.classifier(img_features, text_features)

img_path = "image_test/Image197.jpg"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4874, 0.4810, 0.4684],
                         std=[0.2465, 0.2373, 0.2398])
])
img = Image.open(img_path).convert("RGB")
x_img = transform(img).unsqueeze(0).to(device)

# Fixed text for SHAP
sample_text = "i have lost most stuffs to this unfortunate flood."
encoding = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
x_input_ids = encoding["input_ids"].to(device)
x_attention_mask = encoding["attention_mask"].to(device)

# Pipeline
model_shap = MultiModalPipelineForSHAP(resnet50, bert_model, classifier, x_input_ids, x_attention_mask)

with torch.no_grad():
    logits = model_shap(x_img)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())
    pred_label = class_labels[pred_idx]
    confidence = probs[pred_idx].item()
    print(f"Predicted: {pred_label} (confidence: {confidence:.4f})")

background_img = torch.randn(8, 3, 224, 224).to(device).requires_grad_()
x_img_req = x_img.clone().requires_grad_()
explainer = shap.GradientExplainer(model_shap, background_img)
shap_values = explainer.shap_values(x_img_req, ranked_outputs=1)

if isinstance(shap_values, (tuple, list)):
    shap_values = shap_values[0]

# Sum across channels
shap_values = np.sum(shap_values, axis=1)

x_nhwc = np.transpose(x_img.detach().cpu().numpy(), (0, 2, 3, 1))
x_nhwc = x_nhwc * np.array([0.2465, 0.2373, 0.2398]) + np.array([0.4874, 0.4810, 0.4684])
x_nhwc = np.clip(x_nhwc, 0, 1)

# Use same style as your previous example: label + confidence
shap.image_plot(shap_values, x_nhwc, labels=[f"{pred_label} ({confidence:.2%})"])
plt.show()