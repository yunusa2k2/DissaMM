import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from dataset import CodedImagesDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "coded_images"
csv_file = "Coder1.CSV"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4874, 0.4810, 0.4684],
                         std=[0.2465, 0.2373, 0.2398])
])

dataset = CodedImagesDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

vgg16 = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16.classifier = nn.Identity()  # remove top MLP layers
vgg16 = vgg16.to(device)
vgg16.eval()

feature_vectors = []
labels = []

dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

with torch.no_grad():
    for images, _, _, _, _, damage, _, _ in tqdm(dataloader, desc="Extracting features"):
        #image, image_serial, time_period, relevancy, urgency, damage, relief, text
        images = images.to(device)
        feats = vgg16(images)
        feature_vectors.append(feats.cpu())
        labels.extend(damage.numpy())

labels = torch.tensor(labels, dtype=torch.long)
labels -= labels.min()  # ensure 0-based labels
feature_vectors = torch.cat(feature_vectors)

num_classes = len(torch.unique(labels))
print(f"Feature vector shape: {feature_vectors.shape}, Num classes: {num_classes}")

X_train, X_valid, y_train, y_valid = train_test_split(
    feature_vectors, labels, test_size=0.2, stratify=labels, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels.numpy()),
    y=labels.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

input_dim = feature_vectors.shape[1]

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

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

epochs = 100
patience = 5
best_val_acc = 0.0
patience_counter = 0

for epoch in range(epochs):
    # Training
    mlp.train()
    running_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = mlp(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    mlp.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = mlp(xb)
            _, preds = torch.max(out, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(mlp.state_dict(), "vgg16_mlp_damage.pth")
        print("Model improved & saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("Best validation accuracy:", best_val_acc)