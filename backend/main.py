from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import uvicorn
import os

from model_loader import load_models
from predictor import Predictor

# Global app state to hold instantiated predictor
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting API and loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    # Load models
    model18, model101, classes = load_models(device)
    
    # Initialize predictor
    app_state["predictor"] = Predictor(model18, model101, classes, device)
    
    print("API is ready to accept requests.")
    yield
    print("Shutting down API and cleaning up memory...")
    app_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="CIFAR10 Model Comparison API",
    description="Backend API comparing ResNet18 and ResNet101 predictions for CIFAR10.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for development (React/Next runs on 3000 locally, API on 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for deployment ease; restrict in true production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "CIFAR10 Comparison API is running."}

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": "predictor" in app_state}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image.")
    
    try:
        contents = await file.read()
        predictor = app_state.get("predictor")
        
        if not predictor:
            raise HTTPException(status_code=503, detail="Service unavailable: Models are not loaded yet.")
            
        results = predictor.predict(contents)
        return results
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
