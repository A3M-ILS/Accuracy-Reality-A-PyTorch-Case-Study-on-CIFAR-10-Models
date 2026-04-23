import torch
import torchvision.transforms as T
from PIL import Image
import io

class Predictor:
    def __init__(self, model18, model101, classes, device):
        self.model18 = model18
        self.model101 = model101
        self.classes = classes
        self.device = device
        
        # ResNet18 uses 32x32 images (no resize during training, but handles crops)
        # Assuming inference uses test transform (ToTensor + Normalize)
        # But we resize to 32x32 generally to ensure correct shape just in case user provides larger image
        self.transform18 = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        
        # ResNet101 uses 224x224 images as per training config
        self.transform101 = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    def predict(self, image_bytes: bytes):
        try:
            # We open and convert image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")
            
        x18 = self.transform18(img).unsqueeze(0).to(self.device)
        x101 = self.transform101(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Apply AMP autocast if on CUDA
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits18 = self.model18(x18)
                    logits101 = self.model101(x101)
            else:
                logits18 = self.model18(x18)
                logits101 = self.model101(x101)
                
            probs18 = torch.softmax(logits18, dim=1)[0].tolist()
            probs101 = torch.softmax(logits101, dim=1)[0].tolist()
            
        # Format results
        res18 = [{"className": c, "probability": p} for c, p in zip(self.classes, probs18)]
        res101 = [{"className": c, "probability": p} for c, p in zip(self.classes, probs101)]
        
        # Sort sequentially by probability descending
        res18.sort(key=lambda x: x["probability"], reverse=True)
        res101.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "resnet18": {
                "topPrediction": res18[0]["className"],
                "confidence": res18[0]["probability"],
                "predictions": res18
            },
            "resnet101": {
                "topPrediction": res101[0]["className"],
                "confidence": res101[0]["probability"],
                "predictions": res101
            }
        }
