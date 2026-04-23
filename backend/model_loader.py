import os
import torch
import torch.nn as nn
import torchvision

def disable_inplace_relu(m):
    """Disable inplace ReLU to avoid issues with AMP and certain model architectures."""
    if isinstance(m, nn.ReLU):
        m.inplace = False

def load_models(device):
    """Load both ResNet18 and ResNet101 models into memory."""
    # Define classes based on CIFAR10 training
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    # --- ResNet18 ---
    print("Loading ResNet18...")
    model18 = torchvision.models.resnet18(weights=None)
    model18.fc = nn.Linear(model18.fc.in_features, 10)
    
    # Path configuration - handles both local and docker execution
    path18 = os.environ.get("RESNET18_PATH", "resnet18_cifar10_32.pth")
    if not os.path.exists(path18):
        # Fallback to parent directory for local running without docker
        path18 = os.path.join("..", "resnet18_cifar10_32.pth")
        
    if os.path.exists(path18):
        try:
            ckpt18 = torch.load(path18, map_location=device, weights_only=False)
            if "model_state" in ckpt18:
                model18.load_state_dict(ckpt18["model_state"])
            else:
                model18.load_state_dict(ckpt18)
            print("ResNet18 loaded successfully.")
        except Exception as e:
            print(f"Error loading ResNet18: {e}")
    else:
        print(f"Warning: {path18} not found. ResNet18 will be uninitialized.")

    model18.to(device)
    model18.eval()

    # --- ResNet101 ---
    print("Loading ResNet101...")
    model101 = torchvision.models.resnet101(weights=None)
    model101.fc = nn.Linear(model101.fc.in_features, 10)
    model101.apply(disable_inplace_relu)
    
    path101 = os.environ.get("RESNET101_PATH", "resnet101_cifar10_224.pth")
    if not os.path.exists(path101):
        # Fallback to parent directory for local running without docker
        path101 = os.path.join("..", "resnet101_cifar10_224.pth")
        
    if os.path.exists(path101):
        try:
            ckpt101 = torch.load(path101, map_location=device, weights_only=False)
            if "model_state" in ckpt101:
                model101.load_state_dict(ckpt101["model_state"])
            else:
                model101.load_state_dict(ckpt101)
            print("ResNet101 loaded successfully.")
        except Exception as e:
            print(f"Error loading ResNet101: {e}")
    else:
        print(f"Warning: {path101} not found. ResNet101 will be uninitialized.")

    model101.to(device)
    model101.eval()

    return model18, model101, classes
