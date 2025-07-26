import os
import time
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# -------------------------------
# 1. Custom Dataset Loader (Supports nested folders)
# -------------------------------
class NestedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self._gather_images()

    def _find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _is_image_file(self, filename):
        return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

    def _gather_images(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            for root, _, files in os.walk(class_dir):
                for fname in files:
                    if self._is_image_file(fname):
                        path = os.path.join(root, fname)
                        self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target

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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def binarize_weights(self):
        return torch.sign(self.weight)

    def forward(self, x):
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
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def binarize_weights(self):
        return torch.sign(self.weight)

    def forward(self, x):
        binarized_weights = self.binarize_weights()
        return F.linear(x, binarized_weights, self.bias)

# -------------------------------
# 4. Standard CNN Model
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
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
            nn.MaxPool2d(2, 2),
            BinaryConv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            BinaryConv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
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
# 8. Model Size Functions
# -------------------------------
def get_model_size(model):
    tmp_path = "temp_model.pth"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb

def get_binary_model_size(model):
    total_weight_bits = 0
    total_bias_bytes = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_weight_bits += param.numel()
        elif 'bias' in name:
            total_bias_bytes += param.numel() * 4
    total_bytes = (total_weight_bits / 8) + total_bias_bytes
    size_mb = total_bytes / (1024 * 1024)
    return size_mb

# -------------------------------
# 9. Main Script
# -------------------------------
def main():
    data_dir = "../ModelNet40/ModelNet40_2DTrainTest"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = NestedImageFolder(train_dir, transform=transform)
    test_dataset = NestedImageFolder(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    device = "cpu"
    model_fp32 = SimpleCNN(num_classes)
    model_binary = BinarySimpleCNN(num_classes)

    print("Training FP32 model...")
    train_model(model_fp32, train_loader, device, epochs=5)

    print("Evaluating FP32 model...")
    acc_fp32, lat_fp32 = evaluate_model(model_fp32, test_loader, device)
    size_fp32 = get_model_size(model_fp32)

    print("Applying INT8 dynamic quantization...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32.cpu(), {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

    print("Evaluating INT8 quantized model...")
    acc_int8, lat_int8 = evaluate_model(model_int8, test_loader, device)
    size_int8 = get_model_size(model_int8)

    print("Training fully binary model...")
    train_model(model_binary, train_loader, device, epochs=5)

    print("Evaluating fully binary model...")
    acc_binary, lat_binary = evaluate_model(model_binary, test_loader, device)
    size_binary = get_binary_model_size(model_binary)

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
