import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from dataset import CodedImagesDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from matplotlib.patheffects import withStroke
from scipy.spatial import ConvexHull

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
bert_model.eval()

class MultiModalDataset(CodedImagesDataset):
    def __getitem__(self, idx):
        image, _, _, _, _, _, label, text = super().__getitem__(idx)
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
            'label': torch.tensor(label, dtype=torch.long)
        }

dataset = MultiModalDataset(csv_file, root_dir, transform=image_transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
in_features_img = resnet50.fc.in_features
resnet50.fc = nn.Identity()
for p in resnet50.parameters():
    p.requires_grad = False
resnet50 = resnet50.to(device)
resnet50.eval()

text_feature_dim = bert_model.config.hidden_size

NUM_CLASSES = len(np.unique([dataset[i]['label'].item() for i in range(len(dataset))]))

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

model = MultiModalClassifier(in_features_img, text_feature_dim, NUM_CLASSES).to(device)
model.load_state_dict(torch.load("DissaMM_mlp_relief.pth", map_location=device))
model.eval()

features_list, labels_list, probs_list = [], [], []

with torch.no_grad():
    for batch in loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        img_feat = resnet50(images)
        text_feat = bert_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        concat_feat = torch.cat([img_feat, text_feat], dim=1)

        logits = model(img_feat, text_feat)
        probs = torch.softmax(logits, dim=1)
        pred_probs, pred_classes = torch.max(probs, dim=1)

        features_list.append(concat_feat.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        probs_list.append(pred_probs.cpu().numpy())

features_array = np.vstack(features_list)
labels_array = np.hstack(labels_list)
probs_array = np.hstack(probs_list)

print(f"Features shape: {features_array.shape}, Samples: {len(labels_array)}")

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(features_array)

# class_names = {
#     0: "pre-storm",
#     1: "landfall",
#     2: "Harvey aftermath"
# }

# class_names = {
#     0: "spam/unrelate",
#     1: "not urgent",
#     2: "somewhat urgent",
#     3: "moderately urgent",
#     4: "highly urgent"
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

class_names = {
    0: "not-relief",
    1: "relief",
}

sns.set(style="whitegrid", context="notebook", font_scale=1.2)
palette = sns.color_palette("tab10", n_colors=NUM_CLASSES)

plt.figure(figsize=(14, 12), dpi=600)

for i, lbl in enumerate(np.unique(labels_array)):
    idx = labels_array == lbl
    points = X_umap[idx]
    scores = probs_array[idx]

    plt.scatter(
        points[:, 0], points[:, 1],
        s=scores*300 + 30,
        c=[palette[i]]*len(points),
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5,
        label=class_names[lbl]  # <-- use class name
    )

    if len(points) > 2:
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices,0], points[hull.vertices,1],
                 color=palette[i], alpha=0.1)

    x_mean, y_mean = points.mean(axis=0)
    plt.text(
        x_mean, y_mean, class_names[lbl],  # <-- use class name
        fontsize=16, weight='bold', color=palette[i],
        ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3'),
        path_effects=[withStroke(linewidth=3, foreground="black")]
    )

plt.xlabel("UMAP Dimension 1", fontsize=18, weight='bold')
plt.ylabel("UMAP Dimension 2", fontsize=18, weight='bold')
plt.title("UMAP Visualization of DissaMM Relief Classifier", fontsize=20, weight='bold')
plt.legend(title="Predicted Class", fontsize=14, title_fontsize=14, loc='best')
plt.grid(False)
plt.tight_layout()
plt.show()