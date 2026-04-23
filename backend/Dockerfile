FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies required for OpenCV or Pillow if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .

# Install CPU version of PyTorch to keep image size small for standard deployment
# Models are small enough to run inference on CPU quickly.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app/

# Expose API port
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
