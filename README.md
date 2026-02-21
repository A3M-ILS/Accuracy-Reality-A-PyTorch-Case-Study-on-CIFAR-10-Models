# Accuracy-Reality-A-PyTorch-Case-Study-on-CIFAR-10-Models
From Benchmark to Reality

## Overview

This project explores a common misconception in machine learning:

> High accuracy does not necessarily mean good real-world performance.

Two deep learning models were trained using PyTorch on the CIFAR-10 dataset and evaluated both on the standard test set and on custom real-world images.

The results highlight a gap between benchmark performance and practical usability.

---

## Models Trained

### Model 1: ResNet101 (224×224)

- Input resized from 32×32 to 224×224  
- Deep architecture  
- Test accuracy: ~64%

### Model 2: ResNet18 (32×32)

- Native CIFAR-10 resolution  
- Lightweight architecture  
- Test accuracy: ~86%

---

## Results

| Model     | Test Accuracy | Observed Real-World Behavior           |
|----------|-------------|--------------------------------------|
| ResNet101 | ~64%        | Often performs better on custom images |
| ResNet18  | ~86%        | Struggles more on real-world inputs    |

---

## Key Insight

The model with higher benchmark accuracy (ResNet18) does not necessarily perform better on real-world inputs.

---

## Why This Happens

### 1. Dataset Bias

CIFAR-10 images are:
- Low resolution (32×32)
- Centered objects
- Clean backgrounds

Real-world images are:
- Higher resolution
- More complex scenes
- Variable lighting and viewpoints

---

### 2. Resolution Mismatch

- ResNet18 learns from 32×32 images  
- Real images must be downscaled → loss of information  

ResNet101 operates at 224×224, which better preserves visual details.

---

### 3. Distribution Shift

The training data distribution differs from real-world data.

This mismatch directly impacts generalization.

---

### 4. Small Sample Bias

Manual testing on a few images can be misleading:

- A weaker model may appear better  
- Results are not statistically reliable  

---

## How Accuracy is Computed

Accuracy is defined as:

Accuracy = (Number of correct predictions / Total predictions) × 100

In this project:
- Evaluation is performed on the CIFAR-10 test set (10,000 images)
- This provides a statistically reliable estimate of performance

---

## Tech Stack

- Python  
- PyTorch  
- Torchvision  
- Streamlit (for interface)  
- CUDA (GPU acceleration, RTX 5060)

---

## Demo (Streamlit App)

The project includes a simple interface to test predictions on custom images.

Run locally:

```bash
streamlit run app.py
