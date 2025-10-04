import torch
import torch.nn as nn
from torchvision import models
from ptflops import get_model_complexity_info
import time
import psutil
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG16Model(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Model, self).__init__()
        self.vgg16 = models.vgg16(weights='VGG16_Weights.DEFAULT')
        self.vgg16.classifier = nn.Identity()  # remove original classifier

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1000),
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
        )

    def forward(self, x):
        feats = self.vgg16.features(x)
        feats = torch.flatten(feats, 1)
        return self.mlp(feats)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def benchmark_model(model, dummy_input, precision="fp32", runs=50):
    model.eval()
    dummy_input = dummy_input.to(device)

    # Set precision
    if precision == "fp16":
        model = model.half().to(device)
        dummy_input = dummy_input.half().to(device)
    elif precision == "int8":
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        ).cpu()
        dummy_input = dummy_input.cpu()

    # Warmup
    for _ in range(50):
        _ = model(dummy_input)

    torch.cuda.synchronize() if device.type == "cuda" else None

    # Timing
    start = time.time()
    for _ in range(runs):
        _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    avg_time = (end - start) / runs
    fps = dummy_input.size(0) / avg_time

    # Memory footprint
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 2)

    return avg_time * 1000, fps, mem

num_classes = 10
model = VGG16Model(num_classes=num_classes).to(device)

macs, params = get_model_complexity_info(
    model, (3, 224, 224),
    as_strings=False, print_per_layer_stat=False, verbose=False
)
total_params, trainable_params = count_parameters(model)

print("==== Model Complexity ====")
print(f"Total Params: {total_params/1e6:.2f} M")
print(f"Trainable Params: {trainable_params/1e6:.2f} M")
print(f"FLOPs: {macs/1e9:.2f} GFLOPs")

batch_size = 8
dummy_input = torch.randn(batch_size, 3, 224, 224)

for precision in ["fp32", "fp16", "int8"]:
    try:
        latency, fps, mem = benchmark_model(model, dummy_input, precision)
        print(f"\n[{precision.upper()}]")
        print(f"Avg Latency: {latency:.3f} ms per batch")
        print(f"Throughput: {fps:.2f} images/sec")
        print(f"Memory: {mem:.2f} MB")
    except Exception as e:
        print(f"\n[{precision.upper()}] Not supported: {e}")