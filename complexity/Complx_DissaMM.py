import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel
from ptflops import get_model_complexity_info
import time
import psutil
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DissaMM(nn.Module):
    def __init__(self, img_features, text_features, num_classes):
        super().__init__()
        # MLP head
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

resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet50.fc = nn.Identity()
for p in resnet50.parameters():
    p.requires_grad = False
resnet50 = resnet50.to(device)
resnet50.eval()

bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
for p in bert_model.parameters():
    p.requires_grad = False
bert_model.eval()

img_features = resnet50.fc.in_features if hasattr(resnet50.fc, "in_features") else 2048
text_features = bert_model.config.hidden_size
NUM_CLASSES = 10  # change according to dataset

model = DissaMM(img_features, text_features, NUM_CLASSES).to(device)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def benchmark_model(model, img_feat, text_feat, precision="fp32", runs=50):
    model.eval()
    img_feat, text_feat = img_feat.to(device), text_feat.to(device)

    if precision == "fp16":
        model = model.half().to(device)
        img_feat, text_feat = img_feat.half(), text_feat.half()
    elif precision == "int8":
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8).cpu()
        img_feat, text_feat = img_feat.cpu(), text_feat.cpu()

    # Warmup
    for _ in range(20):
        _ = model(img_feat, text_feat)

    torch.cuda.synchronize() if device.type == "cuda" else None

    start = time.time()
    for _ in range(runs):
        _ = model(img_feat, text_feat)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    avg_time = (end - start) / runs
    fps = img_feat.size(0) / avg_time

    # Memory
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 2)

    return avg_time * 1000, fps, mem

total_params, trainable_params = count_parameters(model)
print("==== DissaMM Complexity ====")
print(f"Total Params: {total_params/1e6:.2f} M")
print(f"Trainable Params: {trainable_params/1e6:.4f} M")

dummy_img_feat = torch.randn(1, img_features)
dummy_text_feat = torch.randn(1, text_features)
combined_feat_dim = img_features + text_features
macs = 2 * combined_feat_dim * 512 + 2 * 512 * 128 + 2 * 128 * NUM_CLASSES  # linear MACs
print(f"Approx FLOPs: {macs/1e6:.2f} M (MLP only)")

batch_size = 8
dummy_img_feat = torch.randn(batch_size, img_features)
dummy_text_feat = torch.randn(batch_size, text_features)

for precision in ["fp32", "fp16", "int8"]:
    try:
        latency, fps, mem = benchmark_model(model, dummy_img_feat, dummy_text_feat, precision)
        print(f"\n[{precision.upper()}]")
        print(f"Avg Latency: {latency:.3f} ms per batch")
        print(f"Throughput: {fps:.2f} images/sec")
        print(f"Memory: {mem:.2f} MB")
    except Exception as e:
        print(f"\n[{precision.upper()}] Not supported: {e}")