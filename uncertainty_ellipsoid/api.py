import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from uncertainty_ellipsoid.config import MODELS_DIR
from uncertainty_ellipsoid.dataset import (
    FeatureCombiner,
    UncertaintySet,
    compute_world_coordinates,
    sample_camera_parameters,
)
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
app = FastAPI(title="Uncertainty Ellipsoid API", lifespan=lifespan)


class PredictionInput(BaseModel):
    pixel_coordinates: List[float] = Field(..., min_items=2, max_items=2)
    depth: float = Field(..., gt=0)
    uncertainty_set: List[float] = Field(..., min_items=20, max_items=20)

    @field_validator("pixel_coordinates")
    def validate_pixel_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError("pixel_coordinates must have exactly 2 elements")
        if not (0 <= v[0] <= 480 and 0 <= v[1] <= 640):
            raise ValueError("pixel_coordinates must be within valid range")
        return v

    @field_validator("depth")
    def validate_depth(cls, v):
        if not (0.2 <= v <= 0.7):
            raise ValueError("depth must be between 0.2 and 0.7")
        return v


class PredictionOutput(BaseModel):
    center: List[float]
    L_matrix: List[List[float]]
    time_ms: float = None


class SimulationInput(BaseModel):
    pixel_coordinates: List[float] = Field(..., min_items=2, max_items=2)
    depth: float = Field(..., gt=0)
    uncertainty_set: List[float] = Field(..., min_items=20, max_items=20)
    num_samples: int = Field(..., gt=0)


class SimulationOutput(BaseModel):
    world_coords: List[List[float]]
    time_ms: float = None


@app.post("/simulate", response_model=SimulationOutput)
async def simulate(input_data: SimulationInput) -> SimulationOutput:
    """Simulate uncertainty ellipsoid for given input using Monte Carlo simulation

    Args:
        input_data (SimulationInput): Input data for simulation

    Returns:
        SimulationOutput: Output data for simulation
    """
    print(input_data)
    start_time = time.perf_counter()
    u, v = input_data.pixel_coordinates
    d = input_data.depth
    uncertainty_set = UncertaintySet(
        f_x=(input_data.uncertainty_set[0], input_data.uncertainty_set[1]),
        f_y=(input_data.uncertainty_set[2], input_data.uncertainty_set[3]),
        c_x=(input_data.uncertainty_set[4], input_data.uncertainty_set[5]),
        c_y=(input_data.uncertainty_set[6], input_data.uncertainty_set[7]),
        rx=(input_data.uncertainty_set[8], input_data.uncertainty_set[9]),
        ry=(input_data.uncertainty_set[10], input_data.uncertainty_set[11]),
        rz=(input_data.uncertainty_set[12], input_data.uncertainty_set[13]),
        tx=(input_data.uncertainty_set[14], input_data.uncertainty_set[15]),
        ty=(input_data.uncertainty_set[16], input_data.uncertainty_set[17]),
        tz=(input_data.uncertainty_set[18], input_data.uncertainty_set[19]),
    )

    M_s = input_data.num_samples

    world_coords = np.array(
        [
            compute_world_coordinates(u, v, d, sample_camera_parameters(uncertainty_set))
            for _ in range(M_s)
        ]
    )
    time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(f"Simulation time: {time_ms:.2f} ms")
    return SimulationOutput(world_coords=world_coords.tolist(), time_ms=time_ms)


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Predict uncertainty ellipsoid for given input
    """
    start_time = time.perf_counter()

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

    prediction_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    logger.info(f"Prediction time: {prediction_time:.2f} ms")

    return PredictionOutput(center=center, L_matrix=L_matrix, time_ms=prediction_time)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
