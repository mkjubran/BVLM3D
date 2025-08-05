import os
import time
import argparse
import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub, prepare, convert, fuse_modules
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import csv
from tqdm import tqdm 


# -------------------------------
# 1. Custom Dataset Loader
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
# 2. Binary Layers
# -------------------------------
class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(BinaryConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

    def binarize_weights(self):
        return torch.sign(self.weight)

    def forward(self, x):
        return F.conv2d(x, self.binarize_weights(), self.bias, stride=self.stride, padding=self.padding)

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    def binarize_weights(self):
        return torch.sign(self.weight)

    def forward(self, x):
        return F.linear(x, self.binarize_weights(), self.bias)

# -------------------------------
# 3. Models
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class BinarySimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(BinarySimpleCNN, self).__init__()
        self.features = nn.Sequential(
            BinaryConv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            BinaryConv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            BinaryConv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(64 * 28 * 28, 128), nn.ReLU(),
            BinaryLinear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class QuantizedSimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(QuantizedSimpleCNN, self).__init__()
        self.quant = QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        return self.dequant(x)

# -------------------------------
# 4. Training and Evaluation
# -------------------------------
def train_model(model, dataloader, device, save_path, epochs=10, lr=1e-3, model_type="fp32"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    log_file = f"../{model_type}_training_log.csv"
    os.makedirs(f"../saved_models/{model_type}", exist_ok=True)

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Model Size (MB)"])

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total
        if model_type == 'fp32':
            size = get_model_size(model)
        elif model_type == 'binary':
            size = get_binary_model_size(model)
        else:
            size = get_int8_model_size(model)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {acc:.2%}, Model Size: {size:.2f} MB")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1,f"{total_loss / len(dataloader):.4f}", f"{acc:.4f}", f"{size:.4f}"])

        if ((epoch + 1) % 5 == 0) or ((epoch <= 10) and ((epoch + 1) % 2)):
            checkpoint_path = f"../saved_models/{model_type}/model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

def train_int8_model(model, dataloader, device, save_path, epochs=10, lr=1e-3):
    model.train()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    fuse_modules(model, [['features.0', 'features.1'], ['features.3', 'features.4'], ['features.6', 'features.7']], inplace=True)
    model = prepare(model, inplace=True)  # Prepare QAT model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    log_file = "../int8_training_log.csv"
    os.makedirs("../saved_models/int8", exist_ok=True)
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Model Size (MB)"])

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total

        # Convert to quantized model for size estimation (copy original to avoid destroying training model)
        model_copy = torch.quantization.convert(model, inplace=False)
        size = get_model_size(model_copy)

        print(f"[INT8] Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {acc:.2%}, Model Size: {size:.2f} MB")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{total_loss / len(dataloader):.4f}", f"{acc:.4f}", f"{size:.4f}"])

        if ((epoch + 1) % 5 == 0) or ((epoch <= 10) and ((epoch + 1) % 2)):
            torch.save(model_copy.state_dict(), f"../saved_models/int8/model_epoch{epoch+1}.pth")
            print(f"Saved INT8 checkpoint at epoch {epoch+1}.")

    # Final conversion and save
    model_final = torch.quantization.convert(model, inplace=False)
    torch.save(model_final.state_dict(), save_path)
    print("Final INT8 model trained and saved.")


def evaluate_model(model, dataloader, device):
    model.eval().to(device)
    all_preds, all_labels = [], []
    start = time.time()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"Evaluate"):
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    latency = (time.time() - start) / len(dataloader.dataset)
    return acc, latency

# -------------------------------
# 5. Size Estimation
# -------------------------------
def get_model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / 1024**2
    os.remove("temp.pth")
    return size

def get_binary_model_size(model):
    weights = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)
    biases = sum(p.numel() * 4 for n, p in model.named_parameters() if 'bias' in n)
    return (weights / 8 + biases) / 1024**2

def get_int8_model_size(model):
    # INT8 quantized model size estimation is the same as FP32 saved model file size
    return get_model_size(model)

# -------------------------------
# 6. Main Logic
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--model_type', choices=['fp32', 'binary', 'int8'], required=True)
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    data_dir = "../ModelNet40/ModelNet40_2DTrainTest"
    #data_dir = "../ModelNet40/ModelNet40_2DTrainTestSome"
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_loader = DataLoader(NestedImageFolder(os.path.join(data_dir, "train"), transform), batch_size=8, shuffle=True)
    test_loader = DataLoader(NestedImageFolder(os.path.join(data_dir, "test"), transform), batch_size=8)
    num_classes = len(train_loader.dataset.classes)

    device = "cpu"
    save_path = f"{args.model_type}_model.pth"

    if args.model_type == 'fp32':
        model = SimpleCNN(num_classes)
    elif args.model_type == 'binary':
        model = BinarySimpleCNN(num_classes)
    elif args.model_type == 'int8':
        model = QuantizedSimpleCNN(num_classes)
    else:
        raise ValueError("Unsupported model type")

    if args.mode == 'train':
        if args.model_type == 'int8':
            train_int8_model(model, train_loader, device, save_path, epochs=args.epochs)
        else:
            train_model(model, train_loader, device, save_path, epochs=args.epochs, model_type=args.model_type)

    else:  # eval
        model.load_state_dict(torch.load(save_path, map_location=device))
        acc, lat = evaluate_model(model, test_loader, device)
        if args.model_type == 'fp32':
            size = get_model_size(model)
        elif args.model_type == 'binary':
            size = get_binary_model_size(model)
        else:
            size = get_int8_model_size(model)
        print(f"{args.model_type.upper()} Model Accuracy: {acc:.2%}, Avg Latency: {lat*1000:.2f} ms, Size: {size:.2f} MB")

if __name__ == "__main__":
    main()
