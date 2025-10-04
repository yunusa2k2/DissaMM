#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DissaMM

@author: yunusa2k2
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from dataset import CodedImagesDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "coded_images"
csv_file = "Coder1.CSV"

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4874, 0.4810, 0.4684],
                         std=[0.2465, 0.2373, 0.2398])
])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()  # freeze BERT

class MultiModalDataset(CodedImagesDataset):
    def __getitem__(self, idx):
        
        image, _, time_period, _, _, _, _, text = super().__getitem__(idx)
        # image, image_serial, time_period, relevancy, urgency, damage, relief, text
        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(time_period, dtype=torch.long)
        }

dataset = MultiModalDataset(csv_file, root_dir, transform=image_transform)

all_labels = [dataset[i]['label'].item() for i in range(len(dataset))]
labels_tensor = torch.tensor(all_labels)
NUM_CLASSES = len(torch.unique(labels_tensor))
print(f"Detected number of classes: {NUM_CLASSES}")

train_indices, val_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=labels_tensor,
    random_state=42
)

train_subset = torch.utils.data.Subset(dataset, train_indices)
val_subset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
in_features_img = resnet50.fc.in_features
resnet50.fc = nn.Identity()  # remove classifier
for param in resnet50.parameters():
    param.requires_grad = False
resnet50 = resnet50.to(device)
resnet50.eval()

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

text_feature_dim = bert_model.config.hidden_size
model = MultiModalClassifier(in_features_img, text_feature_dim, NUM_CLASSES).to(device)

class_weights = compute_class_weight('balanced', classes=np.unique(labels_tensor.numpy()), y=labels_tensor.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

EPOCHS = 50
patience = 5
best_val_acc = 0
counter = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    running_loss = 0
    correct, total = 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            img_features = resnet50(images)
            text_features = bert_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        optimizer.zero_grad()
        outputs = model(img_features, text_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            img_features = resnet50(images)
            text_features = bert_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            outputs = model(img_features, text_features)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "DissaMM_mlp_urgency2.pth")
        print("Model improved & saved.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print("Best validation accuracy:", best_val_acc)