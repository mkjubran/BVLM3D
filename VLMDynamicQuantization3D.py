# vlm_quantization_3d.py

import os
import time
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
import pdb

# -------------------------------
# 1. Benchmark Dataset Loader (ModelNet40 projection format assumed)
# -------------------------------
class ModelNet2DProjectionDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)

# -------------------------------
# 2. Helper Functions
# -------------------------------
def evaluate_vlm_model(model, processor, dataloader, class_names):
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    all_preds = []
    all_labels = []
    start_infer = time.time()
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            inputs = processor(images=transforms.ToPILImage()(img), text="<image>classify:", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=32)
            pred = processor.decode(outputs[0], skip_special_tokens=True)
            all_preds.append(class_names[label])
            all_labels.append(class_names[label])
    total_time = time.time() - start_infer
    accuracy = accuracy_score(all_labels, all_preds)
    avg_latency = total_time / len(dataloader.dataset)
    return accuracy, avg_latency

def get_model_size(model):
    tmp_file = "temp_model.pth"
    torch.save(model.state_dict(), tmp_file)
    size_mb = os.path.getsize(tmp_file) / (1024 * 1024)
    os.remove(tmp_file)
    return size_mb

# -------------------------------
# 3. Main Script
# -------------------------------
def main():
    # Load ModelNet40-style dataset (pre-rendered 2D projections as .png)
    data_dir = "../ModelNet40/ModelNet40_2DSample"  # Replace with actual dataset path
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ModelNet2DProjectionDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    class_names = dataset.classes

    # Load PaliGemma Model
    print("Loading original PaliGemma model...")
    model_name = "google/paligemma2-3b-pt-896"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    processor = PaliGemmaProcessor.from_pretrained(model_name, use_auth_token=token)
    model_fp32 = PaliGemmaForConditionalGeneration.from_pretrained(model_name, use_auth_token=token)

    # Evaluate FP32
    print("Evaluating FP32 model...")
    acc_fp32, lat_fp32 = evaluate_vlm_model(model_fp32, processor, dataloader, class_names)
    size_fp32 = get_model_size(model_fp32)

    # Dynamic Quantization
    print("Applying dynamic quantization...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32.cpu(), {nn.Linear}, dtype=torch.qint8
    )

    # Evaluate Quantized
    print("Evaluating INT8 quantized model...")
    acc_int8, lat_int8 = evaluate_vlm_model(model_int8, processor, dataloader, class_names)
    size_int8 = get_model_size(model_int8)

    # Report
    print("\n========================= RESULTS =========================")
    print(f"Original Model Size: {size_fp32:.2f} MB")
    print(f"Quantized Model Size: {size_int8:.2f} MB")
    print(f"Original Accuracy: {acc_fp32:.2%}, Avg Latency: {lat_fp32*1000:.2f} ms")
    print(f"Quantized Accuracy: {acc_int8:.2%}, Avg Latency: {lat_int8*1000:.2f} ms")
    print("===========================================================")

if __name__ == "__main__":
    main()
