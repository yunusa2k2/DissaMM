import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import CodedImagesDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from scipy.spatial import ConvexHull
from matplotlib.patheffects import withStroke

sns.set(style="whitegrid", context="notebook", font_scale=1.2)
palette = sns.color_palette("tab10")  # Up to 10 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4874, 0.4810, 0.4684],
        std=[0.2465, 0.2373, 0.2398]
    )
])

csv_file = "Coder1.CSV"
root_dir = "coded_images"
dataset = CodedImagesDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

vgg16 = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16.classifier = nn.Identity()  # remove top MLP layers
vgg16 = vgg16.to(device)
vgg16.eval()

input_dim = 25088  # VGG16 feature dimension

class_names = {
    0: "spam/unrelate",
    1: "not urgent",
    2: "somewhat urgent",
    3: "moderately urgent",
    4: "highly urgent"
}

# class_names = {
#     0: "pre-storm",
#     1: "landfall",
#     2: "Harvey aftermath"
# }

# class_names = {
#     0: "irrelevant",
#     1: "relevant",
#     2: "uncertain"
# }

# class_names = {
#     0: "not-damage",
#     1: "damage",
# }

# class_names = {
#     0: "not-relief",
#     1: "relief",
# }

num_classes = len(class_names)

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
mlp.load_state_dict(torch.load("vgg16_mlp_urgency.pth", map_location=device))
mlp.eval()

features_list, pred_labels_list, mlp_scores_list = [], [], []

with torch.no_grad():
    for batch in loader:
        images, _, _, _, _, _, _, _ = batch
        images = images.to(device)
        feats = vgg16(images)
        logits = mlp(feats)
        probs = torch.softmax(logits, dim=1)
        pred_probs, pred_classes = torch.max(probs, dim=1)

        features_list.append(feats.cpu().numpy())
        pred_labels_list.append(pred_classes.cpu().numpy())
        mlp_scores_list.append(pred_probs.cpu().numpy())

features_array = np.vstack(features_list)
pred_labels_array = np.hstack(pred_labels_list)
mlp_scores = np.hstack(mlp_scores_list)

print(f"Features shape: {features_array.shape}, Samples: {len(pred_labels_array)}")
print(f"MLP Scores shape: {mlp_scores.shape}")

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(features_array)

plt.figure(figsize=(14, 12), dpi=600)

for i, lbl in enumerate(np.unique(pred_labels_array)):
    idx = pred_labels_array == lbl
    points = X_umap[idx]
    scores = mlp_scores[idx]
    
    # Scatter points with size and color scaled by confidence
    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=scores * 300 + 30,
        c=[palette[i]] * len(points),
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5,
        label=class_names[lbl]
    )
    
    # Convex hull for each class
    if len(points) > 2:
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                 color=palette[i], alpha=0.1)
    
    # Centroid annotation with visible styling
    x_mean, y_mean = points.mean(axis=0)
    plt.text(
        x_mean, y_mean, class_names[lbl],
        fontsize=18, weight='bold', color=palette[i],
        ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3'),
        path_effects=[withStroke(linewidth=3, foreground="black")]
    )

plt.xlabel("UMAP Dimension 1", fontsize=18, weight='bold')
plt.ylabel("UMAP Dimension 2", fontsize=18, weight='bold')
plt.title("UMAP Visualization of DisasterNet Urgency Classifier (VGG16)", fontsize=20, weight='bold')
plt.legend(title="Predicted Class", fontsize=14, title_fontsize=14, loc='best')
plt.grid(False)
plt.tight_layout()
plt.show()