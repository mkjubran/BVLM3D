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
import copy
import pdb

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
# 2. Quantized Layers (Binary, 2-bit, 4-bit)
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

class TwoBitConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(TwoBitConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
    
    def quantize_2bit_weights(self):
        # 2-bit quantization: 4 levels {-1.5, -0.5, 0.5, 1.5}
        w = self.weight
        w_max = torch.abs(w).max()
        w_normalized = w / (w_max + 1e-8)  # Normalize to [-1, 1]
        
        # Quantize to 4 levels
        w_q = torch.round(w_normalized * 1.5) / 1.5  # Scale to [-1.5, 1.5] and quantize
        w_q = torch.clamp(w_q, -1.5, 1.5)
        
        return w_q * w_max  # Scale back
    
    def forward(self, x):
        return F.conv2d(x, self.quantize_2bit_weights(), self.bias, stride=self.stride, padding=self.padding)

class TwoBitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TwoBitLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
    
    def quantize_2bit_weights(self):
        # 2-bit quantization: 4 levels {-1.5, -0.5, 0.5, 1.5}
        w = self.weight
        w_max = torch.abs(w).max()
        w_normalized = w / (w_max + 1e-8)
        
        w_q = torch.round(w_normalized * 1.5) / 1.5
        w_q = torch.clamp(w_q, -1.5, 1.5)
        
        return w_q * w_max
    
    def forward(self, x):
        return F.linear(x, self.quantize_2bit_weights(), self.bias)

class FourBitConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(FourBitConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
    
    def quantize_4bit_weights(self):
        # 4-bit quantization: 16 levels
        w = self.weight
        w_max = torch.abs(w).max()
        w_normalized = w / (w_max + 1e-8)  # Normalize to [-1, 1]
        
        # Quantize to 16 levels (4-bit): -7/7, -6/7, ..., 6/7, 7/7
        scale = 7.0
        w_q = torch.round(w_normalized * scale) / scale
        w_q = torch.clamp(w_q, -1.0, 1.0)
        
        return w_q * w_max  # Scale back
    
    def forward(self, x):
        return F.conv2d(x, self.quantize_4bit_weights(), self.bias, stride=self.stride, padding=self.padding)

class FourBitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FourBitLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
    
    def quantize_4bit_weights(self):
        # 4-bit quantization: 16 levels
        w = self.weight
        w_max = torch.abs(w).max()
        w_normalized = w / (w_max + 1e-8)
        
        scale = 7.0
        w_q = torch.round(w_normalized * scale) / scale
        w_q = torch.clamp(w_q, -1.0, 1.0)
        
        return w_q * w_max
    
    def forward(self, x):
        return F.linear(x, self.quantize_4bit_weights(), self.bias)

# -------------------------------
# 3. Models
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(4),
            #nn.Conv2d(32, 64, 3, padding=1), 
            #nn.ReLU(inplace=True), 
            #nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 128),
            #nn.Linear(64 * 28 * 28, 128),  
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

class TwoBitSimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(TwoBitSimpleCNN, self).__init__()
        self.features = nn.Sequential(
            TwoBitConv2d(3, 16, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
            TwoBitConv2d(16, 32, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
            TwoBitConv2d(32, 64, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            TwoBitLinear(64 * 28 * 28, 128), 
            nn.ReLU(inplace=True),
            TwoBitLinear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

class FourBitSimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(FourBitSimpleCNN, self).__init__()
        self.features = nn.Sequential(
            FourBitConv2d(3, 16, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
            FourBitConv2d(16, 32, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
            FourBitConv2d(32, 64, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            FourBitLinear(64 * 28 * 28, 128), 
            nn.ReLU(inplace=True),
            FourBitLinear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

class QuantizedSimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(QuantizedSimpleCNN, self).__init__()
        self.quant = QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(4),
            #nn.Conv2d(32, 64, 3, padding=1), 
            #nn.ReLU(), 
            #nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 128), 
            #nn.Linear(64 * 28 * 28, 128), 
            nn.ReLU(),
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
def evaluate_accuracy_only(model, dataloader, device):
    """Quick evaluation function that only returns accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"Eval: "):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return correct / total

def train_model(model, train_dataloader, test_dataloader, device, save_path, epochs=10, lr=1e-3, model_type="fp32"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create directories
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../saved_models", exist_ok=True)
    
    log_file = f"../logs/{model_type}_training_log.csv"
    
    # Initialize CSV with headers including evaluation accuracy
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Eval Accuracy", "Model Size (MB)"])
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_dataloader)
        
        # Calculate model size with debugging
        if model_type == 'fp32':
            size = get_model_size(model)
            print(f"FP32 Model - Theoretical size: {calculate_theoretical_model_size(model, 32):.3f} MB")
        elif model_type == 'binary':
            size = get_binary_model_size(model)
            print(f"Binary Model - Theoretical size: {calculate_theoretical_model_size(model, 1):.3f} MB (weights only)")
        elif model_type == '2bit':
            size = get_2bit_model_size(model)
            print(f"2-bit Model - Theoretical size: {calculate_theoretical_model_size(model, 2):.3f} MB (weights only)")
        elif model_type == '4bit':
            size = get_4bit_model_size(model)
            print(f"4-bit Model - Theoretical size: {calculate_theoretical_model_size(model, 4):.3f} MB (weights only)")
        elif model_type == 'int8':
            size = get_quantized_model_size(model)
            print(f"INT8 Model - Actual quantized size calculated")
        else:
            size = get_model_size(model)  # For dynamic_quant
        
        # Evaluate on test/validation data
        print(f"Evaluating model at epoch {epoch+1}...")
        eval_acc = evaluate_accuracy_only(model, test_dataloader, device)
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2%}, Eval Accuracy: {eval_acc:.2%}, Model Size: {size:.2f} MB")
        
        # Log to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{avg_loss:.4f}", f"{train_acc:.4f}", f"{eval_acc:.4f}", f"{size:.4f}"])
        
        # Save model checkpoints with evaluation accuracy
        if ((epoch + 1) % 5 == 0) or (((epoch + 1) < 10) and ((epoch + 1) % 2 == 0)):
            checkpoint_path = f"../saved_models/{model_type}_model_epoch{epoch+1}_acc{eval_acc:.4f}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Also save evaluation metrics for this checkpoint
            metrics_path = f"../saved_models/{model_type}_model_epoch{epoch+1}_metrics.csv"
            with open(metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Epoch", epoch+1])
                writer.writerow(["Train_Loss", f"{avg_loss:.4f}"])
                writer.writerow(["Train_Accuracy", f"{train_acc:.4f}"])
                writer.writerow(["Eval_Accuracy", f"{eval_acc:.4f}"])
                writer.writerow(["Model_Size_MB", f"{size:.4f}"])
    
    # Final evaluation and save
    final_eval_acc = evaluate_accuracy_only(model, test_dataloader, device)
    final_save_path = f"../saved_models/{model_type}_model_final_acc{final_eval_acc:.4f}.pth"
    torch.save(model.state_dict(), final_save_path)
    
    # Also save to the original path for backward compatibility
    torch.save(model.state_dict(), save_path)
    
    print(f"Final model saved to {final_save_path} with evaluation accuracy: {final_eval_acc:.2%}")

def evaluate_model(model, dataloader, device, warmup_batches=5, desc="Evaluating"):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    
    # Warmup
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            x = x.to(device)
            _ = model(x)
    
    # Actual evaluation with timing
    start = time.time()
    inference_times = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc):
            x = x.to(device)
            
            # Time individual batch inference
            batch_start = time.time()
            preds = model(x).argmax(dim=1).cpu()
            batch_time = time.time() - batch_start
            inference_times.append(batch_time)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    total_time = time.time() - start
    acc = accuracy_score(all_labels, all_preds)
    avg_latency = sum(inference_times) / len(all_preds)  # Per sample latency
    
    return acc, avg_latency, total_time

def evaluate_on_train_data(model, train_loader, device):
    """Evaluate model on training data to compare with training accuracy"""
    print("\n" + "="*50)
    print("TRAINING DATA EVALUATION (for comparison)")
    print("="*50)
    
    # Create a non-shuffled version of training data for consistent evaluation
    train_eval_loader = DataLoader(
        train_loader.dataset, 
        batch_size=train_loader.batch_size, 
        shuffle=False  # Important: no shuffling for consistent evaluation
    )
    
    acc, lat, total_time = evaluate_model(model, train_eval_loader, device, desc="Evaluating on Training Data")
    
    print(f"Training Data Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Training Data Samples: {len(train_eval_loader.dataset)}")
    print("="*50)
    
    return acc

# -------------------------------
# 5. Dynamic Quantization
# -------------------------------
def apply_dynamic_quantization(model):
    """Apply dynamic quantization to a trained FP32 model"""
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv2d}, 
        dtype=torch.qint8
    )
    return quantized_model

# -------------------------------
# 6. Size Estimation
# -------------------------------
def get_model_size(model):
    """Get model size by saving to temporary file"""
    temp_path = "temp_model_size.pth"
    torch.save(model.state_dict(), temp_path)
    size = os.path.getsize(temp_path) / (1024 ** 2)  # MB
    os.remove(temp_path)
    return size

def get_model_size(model):
    """Get model size by saving to temporary file"""
    temp_path = "temp_model_size.pth"
    torch.save(model.state_dict(), temp_path)
    size = os.path.getsize(temp_path) / (1024 ** 2)  # MB
    os.remove(temp_path)
    return size

def get_quantized_model_size(model):
    """Get size of quantized model - calculates actual quantized size"""
    total_size_bytes = 0
    
    # Calculate quantized weights and FP32 biases
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module, 'scale'):
            # Quantized layer - weights are INT8 (1 byte each)
            weight_size_bytes = module.weight().numel() * 1  # INT8 = 1 byte per element
            total_size_bytes += weight_size_bytes
            
            # Scale and zero_point are FP32 (4 bytes each)
            total_size_bytes += 4 + 4  # scale + zero_point
            
            # Bias is FP32 if present
            if hasattr(module, 'bias') and module.bias() is not None:
                bias_size_bytes = module.bias().numel() * 4  # FP32 = 4 bytes per element
                total_size_bytes += bias_size_bytes
                
        elif hasattr(module, 'weight'):
            # Regular FP32 layer in quantized model (shouldn't happen after convert())
            weight_size_bytes = module.weight.numel() * 4  # FP32 = 4 bytes per element
            total_size_bytes += weight_size_bytes
            
            if hasattr(module, 'bias') and module.bias is not None:
                bias_size_bytes = module.bias.numel() * 4
                total_size_bytes += bias_size_bytes
    
    return total_size_bytes / (1024 ** 2)  # Convert to MB

def get_binary_model_size(model):
    """Estimate binary model size (1-bit weights + FP32 biases)"""
    total_bits = 0
    bias_bytes = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_bits += param.numel()  # 1 bit per weight
        elif 'bias' in name:
            bias_bytes += param.numel() * 4  # FP32 biases
    
    weight_bytes = total_bits / 8  # Convert bits to bytes
    total_size = (weight_bytes + bias_bytes) / (1024 ** 2)  # MB
    return total_size

def get_2bit_model_size(model):
    """Estimate 2-bit model size (2-bit weights + FP32 biases + scaling factors)"""
    total_bits = 0
    bias_bytes = 0
    scale_bytes = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_bits += param.numel() * 2  # 2 bits per weight
            scale_bytes += 4  # FP32 scaling factor per weight tensor
        elif 'bias' in name:
            bias_bytes += param.numel() * 4  # FP32 biases
    
    weight_bytes = total_bits / 8  # Convert bits to bytes
    total_size = (weight_bytes + bias_bytes + scale_bytes) / (1024 ** 2)  # MB
    return total_size

def get_4bit_model_size(model):
    """Estimate 4-bit model size (4-bit weights + FP32 biases + scaling factors)"""
    total_bits = 0
    bias_bytes = 0
    scale_bytes = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_bits += param.numel() * 4  # 4 bits per weight
            scale_bytes += 4  # FP32 scaling factor per weight tensor
        elif 'bias' in name:
            bias_bytes += param.numel() * 4  # FP32 biases
    
    weight_bytes = total_bits / 8  # Convert bits to bytes
    total_size = (weight_bytes + bias_bytes + scale_bytes) / (1024 ** 2)  # MB
    return total_size

def calculate_theoretical_model_size(model, precision_bits=32):
    """Calculate theoretical model size for any precision"""
    total_bits = 0
    
    for name, param in model.named_parameters():
        param_bits = param.numel() * precision_bits
        total_bits += param_bits
        print(f"{name}: {param.shape} = {param.numel():,} params = {param_bits/8/1024/1024:.3f} MB")
    
    total_mb = total_bits / 8 / 1024 / 1024
    print(f"Total theoretical size ({precision_bits}-bit): {total_mb:.3f} MB")
    return total_mb

# -------------------------------
# 7. Main Logic
# -------------------------------
def get_num_classes_from_model(model_path):
    """Extract number of classes from saved model state dict"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Look for the final classifier layer - it should be the last layer in classifier
        classifier_layers = {}
        for key in checkpoint.keys():
            if 'classifier.' in key and '.weight' in key:
                # Extract layer number (e.g., 'classifier.3.weight' -> 3)
                layer_num = int(key.split('.')[1])
                layer_weights = checkpoint[key]
                if len(layer_weights.shape) == 2:  # Linear layer
                    classifier_layers[layer_num] = layer_weights.shape
        
        if classifier_layers:
            # Get the highest numbered layer (final layer)
            final_layer_num = max(classifier_layers.keys())
            final_layer_shape = classifier_layers[final_layer_num]
            num_classes = final_layer_shape[0]  # Output features = num_classes
            print(f"Detected final classifier layer: classifier.{final_layer_num} with shape {final_layer_shape}")
            return num_classes
        
        return None
    except Exception as e:
        print(f"Warning: Could not determine number of classes from model: {e}")
        return None

def print_model_architecture(model_path):
    """Print the architecture of saved model for debugging"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"\nModel Architecture from {model_path}:")
        print("-" * 50)
        
        for key, tensor in checkpoint.items():
            if 'classifier' in key:
                print(f"{key}: {tensor.shape}")
        print("-" * 50)
    except Exception as e:
        print(f"Could not load model for architecture inspection: {e}")

def main():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation System")
    parser.add_argument('--mode', choices=['train', 'eval'], required=True,
                       help='Mode: train or evaluate')
    parser.add_argument('--model_type', choices=['fp32', 'dynamic_quant', 'int8', 'binary', '2bit', '4bit'], required=True,
                       help='Model type: fp32, dynamic_quant, int8, binary, 2bit, or 4bit')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model for evaluation. If not provided, uses default naming.')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes. If not provided, inferred from dataset or model.')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default="../ModelNet40/ModelNet40_2DTrainTest",
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training and evaluation')
    parser.add_argument('--debug_model', action='store_true',
                       help='Print model architecture for debugging')
    parser.add_argument('--eval_train', action='store_true',
                       help='Also evaluate on training data for comparison with training accuracy')
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda'],
                       help='Device to use for computation')
    
    args = parser.parse_args()
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Determine number of classes
    if args.mode == 'train':
        # For training, get from dataset
        train_dataset = NestedImageFolder(os.path.join(args.data_dir, "train"), transform)
        test_dataset = NestedImageFolder(os.path.join(args.data_dir, "test"), transform)
        
        if args.num_classes is None:
            num_classes = len(train_dataset.classes)
        else:
            num_classes = args.num_classes
        print(f"Training with {num_classes} classes")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
    else:
        # For evaluation, try to get from model first, then dataset
        model_path_for_classes = args.model_path if args.model_path else f"../saved_models/{args.model_type}_model.pth"
        
        # Special handling for dynamic quantization
        if args.model_type == 'dynamic_quant':
            if args.model_path and 'dynamic_quant' in args.model_path:
                fp32_model_path = args.model_path.replace('dynamic_quant', 'fp32_for_dynamic_quant')
            elif args.model_path:
                fp32_model_path = args.model_path
            else:
                fp32_model_path = "../saved_models/fp32_for_dynamic_quant_model.pth"
            model_path_for_classes = fp32_model_path
        
        if args.num_classes is None:
            if os.path.exists(model_path_for_classes):
                # Debug model architecture if requested
                if args.debug_model:
                    print_model_architecture(model_path_for_classes)
                
                num_classes = get_num_classes_from_model(model_path_for_classes)
                if num_classes is not None:
                    print(f"Detected {num_classes} classes from saved model")
                else:
                    # Fallback to dataset
                    test_dataset = NestedImageFolder(os.path.join(args.data_dir, "test"), transform)
                    num_classes = len(test_dataset.classes)
                    print(f"Using {num_classes} classes from dataset (model detection failed)")
                    if args.debug_model:
                        print_model_architecture(model_path_for_classes)
            else:
                # Fallback to dataset
                test_dataset = NestedImageFolder(os.path.join(args.data_dir, "test"), transform)
                num_classes = len(test_dataset.classes)
                print(f"Using {num_classes} classes from dataset (model not found)")
        else:
            num_classes = args.num_classes
            print(f"Using {num_classes} classes from command line argument")
        
        # Create test loader
        test_loader = DataLoader(
            NestedImageFolder(os.path.join(args.data_dir, "test"), transform), 
            batch_size=args.batch_size
        )
        
        # Also create train loader if eval_train is requested
        if args.eval_train:
            train_dataset = NestedImageFolder(os.path.join(args.data_dir, "train"), transform)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save paths
    os.makedirs("../saved_models", exist_ok=True)
    
    # Use provided model path or default naming
    if args.model_path:
        model_path = args.model_path
        save_path = args.model_path if args.mode == 'train' else args.model_path
    else:
        save_path = f"../saved_models/{args.model_type}_model.pth"
        model_path = save_path
    
    if args.mode == 'train':
        print(f"Training {args.model_type} model...")
        
        if args.model_type == 'fp32':
            model = SimpleCNN(num_classes)
            train_model(model, train_loader, test_loader, device, save_path, epochs=args.epochs, model_type=args.model_type)
            
        elif args.model_type == 'binary':
            model = BinarySimpleCNN(num_classes)
            train_model(model, train_loader, test_loader, device, save_path, epochs=args.epochs, model_type=args.model_type)
            
        elif args.model_type == '2bit':
            model = TwoBitSimpleCNN(num_classes)
            train_model(model, train_loader, test_loader, device, save_path, epochs=args.epochs, model_type=args.model_type)
            
        elif args.model_type == '4bit':
            model = FourBitSimpleCNN(num_classes)
            train_model(model, train_loader, test_loader, device, save_path, epochs=args.epochs, model_type=args.model_type)
            
        elif args.model_type == 'int8':
            model = QuantizedSimpleCNN(num_classes)
            model.train()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Fuse Conv+ReLU layers before quantization
            model = fuse_modules(model, [
                ['features.0', 'features.1'],      # Conv2d(3,16) + ReLU
                ['features.3', 'features.4'],      # Conv2d(16,32) + ReLU
                ['classifier.1', 'classifier.2']   # Linear(32*28*28,128) + ReLU
            ])
            
            model = prepare(model)  # Prepare for QAT
            train_model(model, train_loader, test_loader, device, save_path, epochs=args.epochs, model_type=args.model_type)
            
            # Convert to quantized model
            model.eval()
            model = convert(model)
            torch.save(model.state_dict(), save_path)
            print(f"INT8 QAT model trained and saved to: {save_path}")
            
        elif args.model_type == 'dynamic_quant':
            # For dynamic quantization in training mode, we need to train FP32 first
            print("Training FP32 model for dynamic quantization...")
            fp32_model = SimpleCNN(num_classes)
            fp32_save_path = save_path.replace('dynamic_quant', 'fp32_for_dynamic_quant')
            train_model(fp32_model, train_loader, test_loader, device, fp32_save_path, epochs=args.epochs, model_type="fp32")
            
            print("Applying dynamic quantization...")
            quantized_model = apply_dynamic_quantization(fp32_model)
            torch.save(quantized_model.state_dict(), save_path)
            print(f"Dynamic quantized model saved to: {save_path}")
            
            # Also save number of classes info for future reference
            with open(save_path.replace('.pth', '_info.txt'), 'w') as f:
                f.write(f"num_classes: {num_classes}\n")
                f.write(f"fp32_model_path: {fp32_save_path}\n")
    
    else:  # eval mode
        print(f"Evaluating {args.model_type} model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        
        if args.model_type == 'fp32':
            model = SimpleCNN(num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            acc, lat, total_time = evaluate_model(model, test_loader, device)
            size = get_model_size(model)
            
        elif args.model_type == 'dynamic_quant':
            # For dynamic quantization evaluation, load FP32 model and apply quantization
            if 'dynamic_quant' in model_path:
                # If the provided path is for dynamic quantized model, look for corresponding FP32 model
                fp32_model_path = model_path.replace('dynamic_quant', 'fp32_for_dynamic_quant')
            else:
                # If provided path is directly FP32 model, use it
                fp32_model_path = model_path
            
            if not os.path.exists(fp32_model_path):
                print(f"Error: FP32 model file not found at {fp32_model_path}")
                print("For dynamic quantization, please provide path to the FP32 model or ensure fp32_for_dynamic_quant model exists")
                return
                
            print(f"Loading FP32 model from: {fp32_model_path}")
            fp32_model = SimpleCNN(num_classes)
            fp32_model.load_state_dict(torch.load(fp32_model_path, map_location=device))
            
            print("Applying dynamic quantization...")
            model = apply_dynamic_quantization(fp32_model)
            acc, lat, total_time = evaluate_model(model, test_loader, device)
            size = get_quantized_model_size(model)
            
        elif args.model_type == 'int8':
            model = QuantizedSimpleCNN(num_classes)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model = fuse_modules(model, [
                ['features.0', 'features.1'],      # Conv2d(3,16) + ReLU
                ['features.3', 'features.4'],      # Conv2d(16,32) + ReLU
                ['classifier.1', 'classifier.2']   # Linear(32*28*28,128) + ReLU
            ])
            model = prepare(model)
            model = convert(model)
            model.load_state_dict(torch.load(model_path, map_location=device))
            acc, lat, total_time = evaluate_model(model, test_loader, device)
            size = get_quantized_model_size(model)
            
        elif args.model_type == 'binary':
            model = BinarySimpleCNN(num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            acc, lat, total_time = evaluate_model(model, test_loader, device, desc="Evaluating on Test Data")
            size = get_binary_model_size(model)
            
        elif args.model_type == '2bit':
            model = TwoBitSimpleCNN(num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            acc, lat, total_time = evaluate_model(model, test_loader, device, desc="Evaluating on Test Data")
            size = get_2bit_model_size(model)
            
        elif args.model_type == '4bit':
            model = FourBitSimpleCNN(num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            acc, lat, total_time = evaluate_model(model, test_loader, device, desc="Evaluating on Test Data")
            size = get_4bit_model_size(model)
        
        # Optionally evaluate on training data for comparison
        if args.eval_train:
            train_acc = evaluate_on_train_data(model, train_loader, device)
        
        # Print comprehensive results
        print("\n" + "="*60)
        print(f"{args.model_type.upper()} MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Model Path: {model_path}")
        if args.eval_train:
            print(f"Training Data Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Test Data Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"Average Latency per Sample: {lat*1000:.4f} ms")
        print(f"Total Inference Time: {total_time:.4f} seconds")
        print(f"Model Size: {size:.4f} MB")
        print(f"Throughput: {len(test_loader.dataset)/total_time:.2f} samples/second")
        print("="*60)
        
        # Analysis of training vs test accuracy difference
        if args.eval_train:
            acc_diff = train_acc - acc
            print(f"\nACCURACY ANALYSIS:")
            print(f"Training-Test Accuracy Difference: {acc_diff:.4f} ({acc_diff*100:.2f}%)")
            if acc_diff > 0.05:  # 5% difference
                print("⚠️  Significant difference detected - possible overfitting")
            elif acc_diff > 0.02:  # 2% difference
                print("⚠️  Moderate difference - some overfitting may be present")
            else:
                print("✅ Good generalization - small training-test accuracy gap")
            print("="*60)
        
        # Save results to CSV
        results_file = f"../logs/{args.model_type}_evaluation_results.csv"
        os.makedirs("../logs", exist_ok=True)
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            if args.eval_train:
                writer.writerow(["Model_Type", "Train_Accuracy", "Test_Accuracy", "Accuracy_Diff", "Avg_Latency_ms", "Total_Time_s", "Model_Size_MB", "Throughput_samples_per_s"])
                writer.writerow([args.model_type, f"{train_acc:.4f}", f"{acc:.4f}", f"{train_acc-acc:.4f}", f"{lat*1000:.4f}", f"{total_time:.4f}", f"{size:.4f}", f"{len(test_loader.dataset)/total_time:.2f}"])
            else:
                writer.writerow(["Model_Type", "Test_Accuracy", "Avg_Latency_ms", "Total_Time_s", "Model_Size_MB", "Throughput_samples_per_s"])
                writer.writerow([args.model_type, f"{acc:.4f}", f"{lat*1000:.4f}", f"{total_time:.4f}", f"{size:.4f}", f"{len(test_loader.dataset)/total_time:.2f}"])

if __name__ == "__main__":
    main()
