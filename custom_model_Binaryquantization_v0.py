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
# 2. Custom Binary Convolution Layer
# -------------------------------
class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(BinaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        # Store full-precision weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def binarize_weights(self):
        # Apply sign function to weights for {-1, +1}
        return torch.sign(self.weight)

    def forward(self, x):
        # Binarize weights during forward pass
        binarized_weights = self.binarize_weights()
        return F.conv2d(x, binarized_weights, self.bias, stride=self.stride, padding=self.padding)

# -------------------------------
# 3. Custom Binary Linear Layer
# -------------------------------
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store full-precision weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def binarize_weights(self):
        # Apply sign function to weights for {-1, +1}
        return torch.sign(self.weight)

    def forward(self, x):
        # Binarize weights during forward pass
        binarized_weights = self.binarize_weights()
        return F.linear(x, binarized_weights, self.bias)

# -------------------------------
# 4. Standard CNN Model (for FP32 and INT8)
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
# 5. Fully Binary CNN Model
# -------------------------------
class BinarySimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(BinarySimpleCNN, self).__init__()
        self.features = nn.Sequential(
            BinaryConv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            BinaryConv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            BinaryConv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(64 * 28 * 28, 128),
            nn.ReLU(),
            BinaryLinear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# -------------------------------
# 6. Training Function
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
# 7. Evaluation Function
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
# 8. Get Model Size
# -------------------------------
def get_model_size(model):
    tmp_path = "temp_model.pth"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb

# -------------------------------
# 9. Main Script
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

    # Initialize models
    device = "cpu"  # Quantization typically evaluated on CPU
    model_fp32 = SimpleCNN(num_classes)
    model_binary = BinarySimpleCNN(num_classes)

    # Train FP32 model
    print("Training FP32 model...")
    train_model(model_fp32, dataloader, device, epochs=5)

    # Evaluate FP32 model
    print("Evaluating FP32 model...")
    acc_fp32, lat_fp32 = evaluate_model(model_fp32, dataloader, device)
    size_fp32 = get_model_size(model_fp32)

    # Quantize to INT8 (dynamic quantization for Linear and Conv2d)
    print("Applying INT8 dynamic quantization...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32.cpu(), {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

    # Evaluate INT8 model
    print("Evaluating INT8 quantized model...")
    acc_int8, lat_int8 = evaluate_model(model_int8, dataloader, device)
    size_int8 = get_model_size(model_int8)

    # Train binary model
    print("Training fully binary model...")
    train_model(model_binary, dataloader, device, epochs=5)

    # Evaluate binary model (no additional quantization needed)
    print("Evaluating fully binary model...")
    acc_binary, lat_binary = evaluate_model(model_binary, dataloader, device)
    size_binary = get_model_size(model_binary)

    # Report
    print("\n========================= RESULTS =========================")
    print(f"Original FP32 Model Size: {size_fp32:.2f} MB")
    print(f"INT8 Model Size: {size_int8:.2f} MB")
    print(f"Fully Binary Model Size: {size_binary:.2f} MB")
    print(f"Original FP32 Accuracy: {acc_fp32:.2%}, Avg Latency: {lat_fp32*1000:.2f} ms")
    print(f"INT8 Accuracy: {acc_int8:.2%}, Avg Latency: {lat_int8*1000:.2f} ms")
    print(f"Fully Binary Accuracy: {acc_binary:.2%}, Avg Latency: {lat_binary*1000:.2f} ms")
    print("===========================================================")

if __name__ == "__main__":
    main()
