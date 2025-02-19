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
    features_path: Path = PROCESSED_DATA_DIR / "test_features.h5",
    model_path: Path = MODELS_DIR / "ellipsoid_net_top0_0219_116.pth",
    batch_size: int = 512,  # on one GPU
    num_workers: int = 8,
    device: str = "auto",  # auto-detect MPS, CUDA or CPU
    num_epochs: int = 100,
    loss_weight: list[float] = [10, 10, 1000000],  # center_loss, containment_loss, reg_loss
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
        num_workers=num_workers,
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
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize loss function
    criterion = UncertaintyEllipsoidLoss(
        lambda_center=loss_weight[0], lambda_containment=loss_weight[1], lambda_reg=loss_weight[2]
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = {
            "total": 0.0,
            "center": 0.0,
            "containment": 0.0,
            "regularization": 0.0,
        }
        pbar = tqdm(total=len(dataloader.dataset), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in dataloader:
            inputs = batch["feature"].to(device, non_blocking=True)
            targets = batch["world_coordinates"].to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            centers, L_elements = model(inputs)
            loss, info = criterion(targets, centers, L_elements)
            # logger.debug("Containment loss: {:.4f}", info["loss"]["containment"])

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss = {k: v + info["loss"][k] for k, v in running_loss.items()}

            pbar.update(len(inputs))

        avg_loss = {k: v / len(dataloader) for k, v in running_loss.items()}
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss['total']:.4f} "
            f"(Center: {avg_loss['center']:.4f}, "
            f"Containment: {avg_loss['containment']:.4f}, "
            f"Regularization: {avg_loss['regularization']:.4f})"
        )
        # Write loss to TensorBoard
        writer.add_scalar("Loss/Total", avg_loss["total"], epoch)
        writer.add_scalar("Loss/Center", avg_loss["center"], epoch)
        writer.add_scalar("Loss/Containment", avg_loss["containment"], epoch)
        writer.add_scalar("Loss/Regularization", avg_loss["regularization"], epoch)

        # Check if the model is one of the top 3 best models
        if avg_loss["total"] < best_loss:
            best_loss = avg_loss["total"]
            logger.info(f"Saving model to {model_path}")
            torch.save(model.state_dict(), model_path)

    logger.info("Training complete!")
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    app()
