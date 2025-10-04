import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np

from dataset import CodedImagesDataset

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Dataset setup
# -------------------------
root_dir = "coded_images"
csv_file = "Coder1.CSV"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4874, 0.4810, 0.4684],
                         std=[0.2465, 0.2373, 0.2398])
])

dataset = CodedImagesDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

# -------------------------
# Feature extractor (ResNet50 backbone)
# -------------------------
resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device)
resnet50.eval()

feature_vectors, labels = [], []
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

print("Extracting features with ResNet50...")
with torch.no_grad():
    for images, _, time_period, *_ in tqdm(dataloader, desc="Feature Extraction", unit="batch"):
        images = images.to(device)
        feats = resnet50(images)
        feature_vectors.append(feats.cpu())
        labels.extend(time_period.numpy())

labels = torch.tensor(labels, dtype=torch.long)
labels -= labels.min()
feature_vectors = torch.cat(feature_vectors)

X_train, X_valid, y_train, y_valid = train_test_split(
    feature_vectors, labels, test_size=0.2, stratify=labels, random_state=42
)

valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

input_dim = X_valid.shape[1]
num_classes = len(torch.unique(y_valid))

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
mlp.load_state_dict(torch.load("Dissa_mlp_timeperiod.pth", map_location=device))
mlp.eval()

criterion = nn.CrossEntropyLoss()
all_preds, all_labels = [], []
val_loss = 0.0

print("Evaluating model on validation set...")
with torch.no_grad():
    for xb, yb in tqdm(valid_loader, desc="Evaluation", unit="batch"):
        xb, yb = xb.to(device), yb.to(device)
        out = mlp(xb)
        loss = criterion(out, yb)
        val_loss += loss.item() * xb.size(0)

        _, preds = torch.max(out, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

val_loss /= len(valid_loader.dataset)
all_preds, all_labels = np.array(all_preds), np.array(all_labels)

acc = accuracy_score(all_labels, all_preds)
f1_micro = f1_score(all_labels, all_preds, average="micro")
f1_macro = f1_score(all_labels, all_preds, average="macro")

print("\n==== Evaluation Results ====")
print(f"Validation Loss : {val_loss:.4f}")
print(f"Validation Acc  : {acc:.4f}")
print(f"F1 (Micro)      : {f1_micro:.4f}")
print(f"F1 (Macro)      : {f1_macro:.4f}")