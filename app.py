import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import streamlit as st
from PIL import Image

MODEL_PATH = "resnet18_cifar10_32.pth"

def disable_inplace_relu(m):
    if isinstance(m, nn.ReLU):
        m.inplace = False

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.apply(disable_inplace_relu)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    return model, classes, transform, device

def predict(model, classes, transform, device, image: Image.Image):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(x)
        else:
            logits = model(x)

        probs = torch.softmax(logits, dim=1)[0]
        top3_prob, top3_idx = torch.topk(probs, k=3)

    return [(classes[i], float(p)) for p, i in zip(top3_prob.tolist(), top3_idx.tolist())]

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CIFAR-10 Classifier (ResNet-18, 32×32)")
st.caption("Upload an image and the model will predict one of: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.")

st.write("GPU name:", torch.cuda.get_device_name(0))
# Load model
try:
    model, classes, transform, device = load_model()
    st.success(f"Model loaded on: {device.upper()}")
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_container_width=True)

    if st.button("Predict"):
        top3 = predict(model, classes, transform, device, img)

        st.subheader("Prediction")
        st.write(f"**{top3[0][0]}**  (confidence: **{top3[0][1]:.3f}**)")

        st.subheader("Top-3")
        for label, conf in top3:
            st.write(f"- {label}: {conf:.3f}")