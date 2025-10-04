import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from dataset import CodedImagesDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

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
train_indices, val_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=labels_tensor,
    random_state=42
)
val_subset = torch.utils.data.Subset(dataset, val_indices)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

val_labels = [dataset[i]['label'].item() for i in val_indices]
NUM_CLASSES = len(np.unique(val_labels))
print(f"Detected NUM_CLASSES from validation data: {NUM_CLASSES}")

resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
in_features_img = resnet50.fc.in_features
resnet50.fc = nn.Identity()
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

checkpoint_path = "DissaMM_mlp_time_period.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load weights but allow skipping mismatched layers
model.load_state_dict(checkpoint, strict=False)
model.eval()

criterion = nn.CrossEntropyLoss()

all_preds, all_labels = [], []
val_loss = 0.0
val_iter = tqdm(val_loader, desc="Evaluating", unit="batch")

with torch.no_grad():
    for batch in val_iter:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        img_features = resnet50(images)
        text_features = bert_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        outputs = model(img_features, text_features)

        loss = criterion(outputs, labels)
        val_loss += loss.item() * labels.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        val_iter.set_postfix(val_loss=val_loss/(len(all_preds)))

val_loss /= len(val_subset)
all_preds, all_labels = np.array(all_preds), np.array(all_labels)

acc = accuracy_score(all_labels, all_preds)
f1_micro = f1_score(all_labels, all_preds, average="micro")
f1_macro = f1_score(all_labels, all_preds, average="macro")

print("\n==== Evaluation Results ====")
print(f"Validation Loss : {val_loss:.4f}")
print(f"Validation Acc  : {acc:.4f}")
print(f"F1 (Macro)      : {f1_macro:.4f}")
print(f"F1 (Micro)      : {f1_micro:.4f}")
