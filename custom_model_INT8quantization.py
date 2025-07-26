import os
import time
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# -------------------------------
# 1. Dataset Loader
# -------------------------------
class ModelNet2DProjectionDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)

# -------------------------------
# 2. Custom CNN Model
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# -------------------------------
# 3. Training Function
# -------------------------------
def train_model(model, dataloader, device, epochs=10, lr=1e-3):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# -------------------------------
# 4. Evaluation Function
# -------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_latency = (time.time() - start_time) / len(dataloader.dataset)
    return accuracy, avg_latency

# -------------------------------
# 5. Get Model Size
# -------------------------------
def get_model_size(model):
    tmp_path = "temp_model.pth"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb

# -------------------------------
# 6. Main Script
# -------------------------------
def main():
    data_dir = "../ModelNet40/ModelNet40_2DSample"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ModelNet2DProjectionDataset(data_dir, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Split dataset (for simplicity using all data for train/test here)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model
    model_fp32 = SimpleCNN(num_classes)

    # Train
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training FP32 model...")
    train_model(model_fp32, dataloader, device, epochs=5)

    # Evaluate
    print("Evaluating FP32 model...")
    acc_fp32, lat_fp32 = evaluate_model(model_fp32, dataloader, device)
    size_fp32 = get_model_size(model_fp32)

    # Quantization (Dynamic, for Linear layers)
    print("Applying dynamic quantization...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32.cpu(), {nn.Linear, nn.Conv2d, nn.MaxPool2d}, dtype=torch.qint8
    )

    print("Evaluating INT8 quantized model...")
    acc_int8, lat_int8 = evaluate_model(model_int8, dataloader, "cpu")
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
