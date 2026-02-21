import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

def disable_inplace_relu(m):
    if isinstance(m, nn.ReLU):
        m.inplace = False

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_one.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load("resnet101_cifar10_224.pth", map_location=device)
    classes = ckpt["classes"]

    model = torchvision.models.resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.apply(disable_inplace_relu)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Same preprocessing used in your training (test transform)
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda") if device == "cuda" else torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = probs.max(dim=0)

    print("Prediction:", classes[top_idx.item()])
    print("Confidence:", float(top_prob))

    # Optional: show top-3
    top3_prob, top3_idx = torch.topk(probs, k=3)
    print("\nTop-3:")
    for p, i in zip(top3_prob.tolist(), top3_idx.tolist()):
        print(f"  {classes[i]}: {p:.4f}")

if __name__ == "__main__":
    main()