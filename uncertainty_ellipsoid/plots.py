from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from uncertainty_ellipsoid.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

import matplotlib.pyplot as plt
import numpy as np

def visualize(fig, world_coords, center, L_matrix):
    """Visualize the simulation results using the Cholesky decomposition of the precision matrix.
    
    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure
        world_coords (List[List[float]]): List of world coordinates
        center (List[float]): Center of the ellipsoid
        L_matrix (List[List[float]]): L matrix (lower triangular matrix) of the ellipsoid
    """
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis
    
    # Convert world_coords to numpy array for easy manipulation
    world_coords = np.array(world_coords)
    
    # Plot the world coordinates (points in space)
    ax.scatter(world_coords[:, 0], world_coords[:, 1], world_coords[:, 2], c='b', marker='o', label="World Coordinates")
    
    # Plot the center of the ellipsoid
    ax.scatter(center[0], center[1], center[2], c='r', marker='X', s=100, label="Ellipsoid Center")
    
    # Create the ellipsoid using parametric equations
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # Parametric equations for a unit sphere
    x = np.outer(np.sin(v), np.cos(u))
    y = np.outer(np.sin(v), np.sin(u))
    z = np.outer(np.cos(v), np.ones_like(u))

    # Apply the L_matrix transformation for the ellipsoid (rotation/scaling)
    ellipsoid_points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
    transformed_coords = np.dot(L_matrix, ellipsoid_points)
    x_transformed, y_transformed, z_transformed = transformed_coords
    
    # Add the center to each point to correctly position the ellipsoid
    x_transformed += center[0]
    y_transformed += center[1]
    z_transformed += center[2]
    
    # Plot the ellipsoid surface
    ax.plot_surface(x_transformed.reshape(x.shape), y_transformed.reshape(y.shape), z_transformed.reshape(z.shape), 
                    color='g', alpha=0.3, rstride=5, cstride=5, label="Ellipsoid Surface")
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a legend
    ax.legend()
    
    # Show the plot
    plt.show()





@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
