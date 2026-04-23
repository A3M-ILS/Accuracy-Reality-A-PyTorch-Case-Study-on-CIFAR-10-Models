# Accuracy ≠ Reality: A PyTorch Case Study on CIFAR-10 Models

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red?logo=pytorch)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

---

## The Question

> Does higher benchmark accuracy mean better real-world performance?

This project challenges that assumption by training two ResNet models on CIFAR-10 and evaluating them side-by-side — both on the standard test set and on custom real-world images.

The answer is no. And this app shows exactly why.

---

## What Was Found

Two models were trained on the same dataset, with different architectures and input resolutions:

| Model | Architecture | Input Size | Test Accuracy |
|-------|-------------|-----------|---------------|
| ResNet18 | 18 layers, 11.1M params | 32×32 (native) | ~86% |
| ResNet101 | 101 layers, 42.5M params | 224×224 (upscaled) | ~64% |

**ResNet18 scores 22 percentage points higher on the benchmark.**

Yet when tested on real-world images, ResNet101 frequently outperforms it. The model with the lower accuracy number produces more sensible predictions in practice.

---

## Why This Happens

### 1. Dataset Bias

CIFAR-10 images are artifacts of a benchmark, not the real world:

| CIFAR-10 Images | Real-World Images |
|----------------|------------------|
| 32×32 pixels | High resolution |
| Centered objects | Complex, cluttered scenes |
| Clean, uniform backgrounds | Variable lighting and viewpoints |
| Artificially balanced classes | Naturally skewed distributions |

ResNet18 mastered this specific distribution. Outside of it, that mastery breaks down.

### 2. Resolution Mismatch

ResNet18 was trained on 32×32 images. When a real-world photo is downscaled to 32×32 for inference, most of the visual information is destroyed before the model even sees it.

ResNet101 operates at 224×224 — the resolution real images are taken at. It sees what you see.

### 3. Distribution Shift

Benchmark accuracy measures performance on data drawn from the same distribution as training data. Real-world images come from a fundamentally different distribution. A model that overfits to benchmark characteristics will fail to generalize — regardless of its accuracy score.

### 4. Accuracy as a Single Number is Lossy

Accuracy = correct predictions / total predictions. It compresses the entire behavior of a model into one number, discarding everything about *how* it fails, *where* it struggles, and *what* it confuses. Two models with identical accuracy scores can have completely different failure modes.

---

## The Interface

A full-stack web application was built to make this comparison interactive and immediate.

Upload any image and watch both models run inference in real time. The UI shows:
- **Top prediction and confidence** for each model
- **Full class probability distribution** with animated bars
- **Side-by-side winner highlight** based on confidence
- **Explanation panel** on why the models agree or disagree

![Preview](https://via.placeholder.com/900x450?text=ResNet18+vs+ResNet101+%E2%80%94+Live+Comparison+UI)

---

## Tech Stack

**ML & Training**
- PyTorch 2.2.1, Torchvision — model definition, training, inference
- CUDA / AMP — mixed-precision GPU training (RTX 5060)
- CIFAR-10 dataset via `torchvision.datasets`

**Backend**
- FastAPI — async REST API with `/predict` endpoint
- Uvicorn — ASGI server
- Pillow — image preprocessing

**Frontend**
- Next.js 14 (App Router) — React framework
- Tailwind CSS — utility-first styling with glassmorphism design
- Framer Motion — animated probability bars and transitions
- Lucide React — icons

**Infrastructure**
- Docker + Docker Compose — containerized full-stack deployment

---

## Running Locally

### With Docker (recommended)

Make sure the trained model weights are in the project root:
- `resnet18_cifar10_32.pth`
- `resnet101_cifar10_224.pth`

```bash
docker-compose up --build
```

Open `http://localhost:3000`.

### Without Docker

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

---

## Training the Models

Both training scripts are included. A CUDA-capable GPU is required.

```bash
# ResNet18 — native 32x32, 50 epochs
python train_resnet18.py

# ResNet101 — upscaled to 224x224, 10 epochs
python train_resnet101.py
```

Checkpoints are saved as `.pth` files in the project root.

---

## Deploying to the Cloud

The frontend and backend are decoupled and can be deployed independently.

**Backend** (Render, Railway, Fly.io):
- Use Docker environment
- Set root directory to `backend/`
- Model weights must be available at runtime — use a storage bucket or Git LFS if deploying via CI/CD

**Frontend** (Vercel, Netlify, Render):
- Set root directory to `frontend/`
- Set the environment variable:
  ```
  NEXT_PUBLIC_API_URL=https://your-backend-url.com
  ```

---

## Takeaway

Accuracy benchmarks are a useful signal, not a verdict. Evaluating a model means understanding its failure modes, its training distribution, and what happens when that distribution shifts. This project is a small but concrete demonstration of that gap.
