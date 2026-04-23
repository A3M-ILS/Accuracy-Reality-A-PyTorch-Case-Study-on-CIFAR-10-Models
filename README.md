# CIFAR10 Model Battle: ResNet18 vs ResNet101

![Deploy Status](https://img.shields.io/badge/Deploy-Ready-green)
![Live Demo](https://img.shields.io/badge/Live%20Demo-[link%20placeholder]-blue)

A complete, production-ready full-stack web application designed to evaluate and compare PyTorch ResNet models concurrently on the CIFAR10 dataset.

### High-Fidelity UI Interface
It incorporates dynamic probability bars side-by-side, high-end "glassmorphism" UI design built in Tailwind CSS, and polished animations using Framer Motion. 
When predictions differ, the interface identifies and explains precisely how architectural differences affect deep feature representations.

![Preview](https://via.placeholder.com/800x400?text=App+Preview+Placeholder)

---

## 🚀 Features
- **Concurrent Inference**: Evaluates ResNet18 (native 32x32 images) and ResNet101 (upscaled 224x224 images) concurrently.
- **Dynamic Head-to-Head Comparison**: Highlights highest-confidence predictions in real-time.
- **Glassmorphism UI**: High-end UX using Next.js, Framer Motion, and Tailwind CSS.
- **Async Python Backend**: FastAPI implementation minimizing RAM footprints and ensuring low-latency CORS compatibility.
- **Explainability module**: Exposes reasoning on why different models produce different categorisations.
- **Production-Ready Docker Config**: Deploy universally locally or on Platform-as-a-Service environments natively.

---

## 💻 Tech Stack
- Frontend: Next.js App Router, React 18, Tailwind CSS, Framer Motion.
- Backend: FastAPI, PyTorch CPU, Uvicorn, Pillow.
- Infrastructure: Docker, Docker Compose.

---

## ✨ Local Development & Execution

Ensure you have **Docker** and **Docker Compose** installed.

1. Ensure your trained models are located in the root of the project:
   - `resnet18_cifar10_32.pth`
   - `resnet101_cifar10_224.pth`
2. Bring up the docker stack:
   ```bash
   docker-compose up --build
   ```
3. Open `http://localhost:3000` to interact with the application.

---

## 🌍 Platform Deployments (Render, Railway, Hugging Face Spaces)

This code is partitioned natively to allow decoupled frontend and backend deployment.

**1. Creating the Backend Service:**
- Use `Docker` environment
- Set Root Directory to `backend/`
- ***Note:*** Platforms running from GitHub require `.pth` files. If deploying via CI/CD, attach an external storage bucket, use large file storage (LFS), or embed them in your repository natively (if sizes permit).

**2. Creating the Frontend Service:**
- Use Native Node/Next.js environment or `Docker`
- Set Root Directory to `frontend/`
- Set `NEXT_PUBLIC_API_URL` to the publicly deployed backend route (e.g. `https://my-fastapi.onrender.com`).
