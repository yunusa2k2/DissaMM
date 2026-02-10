#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP t-SNE on COCO Captions with 4 semantic caption groupings
"""

import torch
import clip
from PIL import Image
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import os

# ---------------------------------------------------------
# 1. Semantic caption grouping functions
# ---------------------------------------------------------

def caption_group1(caption):
    caption = caption.lower()

    if any(w in caption for w in ["person", "man", "woman", "child", "people"]):
        return "human"
    elif any(w in caption for w in ["dog", "cat", "horse", "animal", "bird"]):
        return "animal"
    elif any(w in caption for w in ["car", "truck", "bus", "bicycle", "vehicle"]):
        return "vehicle"
    elif any(w in caption for w in ["tree", "grass", "mountain", "river"]):
        return "nature"
    elif any(w in caption for w in ["building", "city", "street", "bridge"]):
        return "urban"
    else:
        return "other"
    

def caption_group2(caption):
    caption = caption.lower()

    if any(w in caption for w in ["kitchen", "bedroom", "indoor", "room"]):
        return "indoor"
    elif any(w in caption for w in ["park", "beach", "street", "outdoor"]):
        return "outdoor"
    elif any(w in caption for w in ["person", "man", "woman", "child"]):
        return "human"
    elif any(w in caption for w in ["dog", "cat", "horse", "animal"]):
        return "animal"
    elif any(w in caption for w in ["car", "truck", "bus", "bike"]):
        return "vehicle"
    else:
        return "other"
    


def caption_group3(caption):
    caption = caption.lower()

    if any(w in caption for w in ["person", "man", "woman", "child"]):
        return "person"
    elif any(w in caption for w in ["chair", "table", "car", "truck", "object"]):
        return "object"
    elif any(w in caption for w in ["tree", "grass", "mountain"]):
        return "nature"
    elif any(w in caption for w in ["dog", "cat", "bird", "animal"]):
        return "animal"
    elif any(w in caption for w in ["sky", "cloud", "sunset"]):
        return "sky"
    else:
        return "other"
    
    

def caption_group4(caption):
    caption = caption.lower()

    if any(w in caption for w in ["running", "walking", "sitting", "playing"]):
        return "person-action"
    elif any(w in caption for w in ["jumping", "sleeping", "flying"]):
        return "animal-action"
    elif any(w in caption for w in ["driving", "riding"]):
        return "vehicle-action"
    elif any(w in caption for w in ["cooking", "cleaning"]):
        return "indoor-action"
    elif any(w in caption for w in ["swimming", "surfing"]):
        return "water-action"
    else:
        return "other"
    
    
def caption_group5(caption):
    caption = caption.lower()

    if any(w in caption for w in ["running", "walking", "playing", "working"]):
        return "activity"
    elif any(w in caption for w in ["park", "kitchen", "street", "beach"]):
        return "place"
    elif any(w in caption for w in ["car", "chair", "table"]):
        return "object"
    elif any(w in caption for w in ["person", "man", "woman", "dog", "cat"]):
        return "living"
    elif any(w in caption for w in ["tree", "mountain", "sky", "river"]):
        return "nature"
    else:
        return "other"
    
def caption_group6(caption):
    caption = caption.lower()

    if any(w in caption for w in ["person", "man", "woman", "child"]):
        return "people"
    elif any(w in caption for w in ["dog", "cat", "animal"]):
        return "animals"
    elif any(w in caption for w in ["car", "truck", "vehicle"]):
        return "vehicles"
    elif any(w in caption for w in ["building", "house"]):
        return "architecture"
    elif any(w in caption for w in ["tree", "grass", "river", "mountain"]):
        return "natural-scenes"
    else:
        return "other"



# ---------------------------------------------------------
# 2. COCO Dataset Loader
# ---------------------------------------------------------

class COCOLoader(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        # Load first caption for the image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]["caption"]

        # Placeholder category (will assign semantic groups later)
        cat = "unknown"

        # Load image
        img_file = self.coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption, cat

# ---------------------------------------------------------
# 3. Create DataLoader
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

coco_loader = DataLoader(
    COCOLoader(
        img_dir="val2017/val2017",
        ann_file="annotations_trainval2017/annotations/captions_val2017.json",
        transform=transform
    ),
    batch_size=1,
    shuffle=False
)

# ---------------------------------------------------------
# 4. Load CLIP
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# ---------------------------------------------------------
# 5. Helper function to compute embeddings for a given caption group
# ---------------------------------------------------------


def compute_embeddings(coco_loader, caption_group_func):
    embeddings = []
    labels = []

    for image, caption, _ in coco_loader:
        # Convert tensor -> PIL -> preprocess
        image_pil = transforms.ToPILImage()(image.squeeze(0))
        image_input = preprocess(image_pil).unsqueeze(0).to(device)

        # Ensure caption is string
        caption_str = caption[0] if isinstance(caption, (tuple, list)) else str(caption)

        # Assign semantic group
        cat_group = caption_group_func(caption_str)
        if cat_group == "other":
            continue  # skip

        text_input = clip.tokenize([caption_str]).to(device)

        with torch.no_grad():
            img_emb = model.encode_image(image_input)
            txt_emb = model.encode_text(text_input)

        fused = (img_emb + txt_emb) / 2
        embeddings.append(fused.cpu().numpy()[0])
        labels.append(cat_group)

    return np.array(embeddings), labels


def plot_tsne(embeddings, labels, title):
    if len(embeddings) == 0:
        print(f"[skip] no embeddings for: {title}")
        return

    z = TSNE(n_components=2, metric='cosine', perplexity=30, random_state=42)\
        .fit_transform(embeddings)

    unique_labels = sorted(list(set(labels)))
    color_map = {lab: idx for idx, lab in enumerate(unique_labels)}

    cmap = plt.get_cmap("tab10")
    colors = [cmap(color_map[l] % cmap.N) for l in labels]

    plt.figure(figsize=(10, 8), dpi=330)
    plt.scatter(z[:, 0], z[:, 1], c=colors, s=12, alpha=0.9)

    # Legend with bigger 16 pt font
    patches = [
        mpatches.Patch(color=cmap(color_map[lab] % cmap.N), label=lab)
        for lab in unique_labels
    ]
    plt.legend(handles=patches, markerscale=3, fontsize=18, frameon=True)

    # Title with 18 pt font
    plt.title(title, fontsize=18)

    # Tick numbering 14 pt
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.show()    

# ---------------------------------------------------------
# 7. Run t-SNE for all 4 caption groups
# ---------------------------------------------------------
group_funcs = [caption_group1, caption_group2, caption_group3, caption_group4, caption_group5, caption_group6]
titles = [
    "t-SNE: Caption Group A",
    "t-SNE: Caption Group B",
    "t-SNE: Caption Group C",
    "t-SNE: Caption Group D",
    "t-SNE: Caption Group E",
    "t-SNE: Caption Group F"
]

for func, title in zip(group_funcs, titles):
    embeddings, labels = compute_embeddings(coco_loader, func)
    plot_tsne(embeddings, labels, title)