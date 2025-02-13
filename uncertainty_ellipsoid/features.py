from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from uncertainty_ellipsoid.config import PROCESSED_DATA_DIR

app = typer.Typer()


def print_hdf5_structure(h5file):
    """
    Recursively print the structure of the HDF5 file.

    Args:
        h5file: The opened HDF5 file object.
    """

    def print_group(name, obj):
        if isinstance(obj, h5py.Group):
            logger.info(f"Group: {name}")
            for key in obj.keys():
                print_group(f"{name}/{key}", obj[key])  # Recursively print sub-groups
        elif isinstance(obj, h5py.Dataset):
            logger.info(f"  Dataset: {name} - Shape: {obj.shape} - Dtype: {obj.dtype}")

    h5file.visititems(print_group)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.h5",
    output_dir: Path = PROCESSED_DATA_DIR,
    test_size: float = typer.Option(0.2, help="测试集比例"),
    random_seed: int = typer.Option(42, help="随机种子"),
):
    """生成分离的训练测试特征文件"""
    try:
        logger.info(f"从 {input_path} 加载数据集")

        with h5py.File(input_path, "r") as f:
            # Print the structure of the HDF5 file
            print_hdf5_structure(f)

            # Load the original data
            world_coordinates = f["world_coordinates"][:]  # [N, M_s, 3]
            pixel_coordinates = f["pixel_coordinates"][:]  # [N, 2]
            depths = f["depths"][:]  # [N, 1]
            uncertainty_sets = f["uncertainty_sets"][:]  # [N, 20]

        # Split the dataset into training and testing sets
        logger.info("分割训练集和测试集")
        train_indices, test_indices = train_test_split(
            np.arange(world_coordinates.shape[0]),
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
        )

        # Save the training features
        train_path = output_dir / "train_features.h5"
        with h5py.File(train_path, "w") as f_train:
            f_train.create_dataset(
                "world_coordinates",
                data=world_coordinates[train_indices],
                compression="gzip",
                chunks=True,
            )
            f_train.create_dataset(
                "pixel_coordinates",
                data=pixel_coordinates[train_indices],
                compression="gzip",
                chunks=True,
            )
            f_train.create_dataset(
                "depths", data=depths[train_indices], compression="gzip", chunks=True
            )
            f_train.create_dataset(
                "uncertainty_sets",
                data=uncertainty_sets[train_indices],
                compression="gzip",
                chunks=True,
            )
            f_train.attrs.update(
                {"samples": len(train_indices), "creation_date": datetime.now().isoformat()}
            )

        # Save the testing features
        test_path = output_dir / "test_features.h5"
        with h5py.File(test_path, "w") as f_test:
            f_test.create_dataset(
                "world_coordinates",
                data=world_coordinates[test_indices],
                compression="gzip",
                chunks=True,
            )
            f_test.create_dataset(
                "pixel_coordinates",
                data=pixel_coordinates[test_indices],
                compression="gzip",
                chunks=True,
            )
            f_test.create_dataset(
                "depths", data=depths[test_indices], compression="gzip", chunks=True
            )
            f_test.create_dataset(
                "uncertainty_sets",
                data=uncertainty_sets[test_indices],
                compression="gzip",
                chunks=True,
            )
            f_test.attrs.update(
                {"samples": len(test_indices), "creation_date": datetime.now().isoformat()}
            )

        logger.success(f"文件保存完成: \n- {train_path}\n- {test_path}")

        # Print the structure of the saved training features file
        logger.info("打印训练集文件结构:")
        with h5py.File(train_path, "r") as f_train:
            print_hdf5_structure(f_train)

        # Print the structure of the saved testing features file
        logger.info("打印测试集文件结构:")
        with h5py.File(test_path, "r") as f_test:
            print_hdf5_structure(f_test)

    except Exception as e:
        logger.error(f"文件保存失败: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
