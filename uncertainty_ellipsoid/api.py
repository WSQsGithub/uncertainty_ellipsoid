from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from uncertainty_ellipsoid.config import MODELS_DIR
from uncertainty_ellipsoid.dataset import FeatureCombiner
from uncertainty_ellipsoid.modeling.model import safe_load_model

# Initialize global variables
model = None
device = None
transform = FeatureCombiner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Startup
    global model, device
    
    # Initialize device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")

    # Load model
    model_path = MODELS_DIR / "ellipsoid_net_top0.pth"
    model = safe_load_model(model_path, device)
    model.eval()
    logger.info("Model loaded successfully")
    
    yield
    
    # Cleanup (if needed)
    # Add any cleanup code here


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Uncertainty Ellipsoid API",
    lifespan=lifespan
)


class PredictionInput(BaseModel):
    pixel_coordinates: List[float] = Field(..., min_items=2, max_items=2)
    depth: float = Field(..., gt=0)
    uncertainty_set: List[float] = Field(..., min_items=20, max_items=20)

    @field_validator('pixel_coordinates')
    def validate_pixel_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError('pixel_coordinates must have exactly 2 elements')
        if not (0 <= v[0] <= 480 and 0 <= v[1] <= 640):
            raise ValueError('pixel_coordinates must be within valid range')
        return v

    @field_validator('depth')
    def validate_depth(cls, v):
        if not (0.2 <= v <= 0.7):
            raise ValueError('depth must be between 0.2 and 0.7')
        return v


class PredictionOutput(BaseModel):
    center: List[float]
    L_matrix: List[List[float]]


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict uncertainty ellipsoid for given input
    """
    # Convert input to tensors
    sample = {
        "pixel_coordinates": torch.tensor(input_data.pixel_coordinates).float(),
        "depth": torch.tensor(input_data.depth).float(),
        "uncertainty_set": torch.tensor(input_data.uncertainty_set).float(),
    }

    # Apply transform
    sample = transform(sample)
    feature = sample["feature"].unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        centers, L_elements = model(feature)
        
        # Convert to numpy for response
        center = centers[0].cpu().numpy().tolist()
        L_matrix = L_elements[0].cpu().numpy().tolist()

        # Replace NaN with 0.0 (or any other value)
        center = np.nan_to_num(center, nan=0.0).tolist()
        L_matrix = np.nan_to_num(L_matrix, nan=0.0).tolist()



    return PredictionOutput(
        center=center,
        L_matrix=L_matrix
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

