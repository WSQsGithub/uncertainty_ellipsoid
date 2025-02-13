import platform
from pathlib import Path

import torch
import typer
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from uncertainty_ellipsoid.config import MODELS_DIR, PROCESSED_DATA_DIR
from uncertainty_ellipsoid.dataset import FeatureCombiner, get_dataloader
from uncertainty_ellipsoid.modeling.loss import UncertaintyEllipsoidLoss
from uncertainty_ellipsoid.modeling.model import safe_load_model

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "train_features.h5",
    model_path: Path = MODELS_DIR / "ellipsoid_net_top1.pth",
    batch_size: int = 64,
    device: str = "auto",  # auto-detect MPS, CUDA or CPU
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

    # If using multiple GPUs, use DataParallel
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initialize loss function
    criterion = UncertaintyEllipsoidLoss()

    # Initialize TensorBoard writer
    writer = SummaryWriter()

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

            running_loss += loss.item()/1000

            pbar.update(len(inputs))

        avg_loss = running_loss / len(dataloader) * 1000
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Write loss to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Check if the model is one of the top 3 best models
        if epoch < 3:
            torch.save(model.state_dict(), MODELS_DIR / f"ellipsoid_net_top{epoch}.pth")
        else:
            # Check if the current model is better than the worst top 3 model
            top_models = sorted(
                MODELS_DIR.glob("ellipsoid_net_top*.pth"), key=lambda x: x.stat().st_mtime
            )
            worst_top_model = top_models[0]
            worst_top_model_loss = float("inf")
            for top_model in top_models:
                model = safe_load_model(top_model, device)
                model.eval()
                running_loss = 0.0
                with torch.no_grad():
                    for batch in dataloader:
                        inputs = batch["feature"].to(device, non_blocking=True)
                        targets = batch["world_coordinates"].to(device, non_blocking=True)

                        centers, L_elements = model(inputs)
                        loss = criterion(targets, centers, L_elements)

                        running_loss += loss.item()

                avg_loss = running_loss / len(dataloader)
                if avg_loss < worst_top_model_loss:
                    worst_top_model = top_model
                    worst_top_model_loss = avg_loss

            # Replace the worst top model with the current model
            if avg_loss < worst_top_model_loss:
                torch.save(model.state_dict(), worst_top_model)

    logger.info("Training complete!")
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    app()
