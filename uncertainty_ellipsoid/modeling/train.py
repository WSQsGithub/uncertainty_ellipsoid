from pathlib import Path

import pandas as pd
import torch
import typer
from loguru import logger
from tqdm import tqdm
import platform

from uncertainty_ellipsoid.config import MODELS_DIR, PROCESSED_DATA_DIR
from uncertainty_ellipsoid.dataset import FeatureCombiner, get_dataloader
from uncertainty_ellipsoid.modeling.model import safe_load_model
from uncertainty_ellipsoid.modeling.loss import UncertaintyEllipsoidLoss

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "train_features.h5",
    model_path: Path = MODELS_DIR / "ellipsoid_net.pth",
    batch_size: int = 64,
    device: str = "auto", # auto-detect MPS, CUDA or CPU
    # -----------------------------------------
):
    """
    Execute model training, automatically handling model initialization

    Args:
        features_path: Path to preprocessed feature data
        model_path: Path to model weights file
        batch_size: Training batch size
        device: Compute device (cuda/cpu)
    """
    # Initialize device
    if device == "auto":
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    device = torch.device(device)
    logger.info(f"Using compute device: {device}")

    # Initialize data pipeline
    logger.info("Preparing dataloader...")
    dataloader = get_dataloader(
        h5_path=features_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        transform=FeatureCombiner(),
    )

    # Initialize model
    logger.info(f"Safe loading model from {model_path} to {device}")
    model = safe_load_model(model_path, device)

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initialize loss function
    criterion = UncertaintyEllipsoidLoss()

    # Training loop
    logger.info("Starting training...")
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(total=len(dataloader.dataset), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in dataloader:
            inputs = batch["feature"].to(device, non_blocking=True)
            targets = batch["world_coordinates"].to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            centers, L_elements = model(inputs)            
            loss = criterion(targets, centers, L_elements)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.update(len(inputs))

        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), MODELS_DIR / f"ellipsoid_net_{epoch}.pth")

    logger.info("Training complete!")

if __name__ == "__main__":
    app()
