import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Tuple

import h5pickle
import h5py
import numpy as np
import torch
import typer
from loguru import logger
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from uncertainty_ellipsoid.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_dataloader(
    h5_path: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    sampler=None,
) -> DataLoader:
    """
    创建数据加载器

    Args:
        h5_path (Path): HDF5数据文件路径
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载的工作进程数
        transform (callable, optional): 数据转换函数

    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = UncertaintyEllipsoidDataset(h5_path, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # 如果使用GPU，启用pin_memory
        drop_last=True,  # 丢弃最后一个不完整的批次
        sampler=sampler,
    )


class FeatureCombiner:
    """数据转换类，将特征和标签组合为输入张量"""

    def __call__(self, sample):
        # 从样本中提取特征和标签
        pixel_coordinates = sample["pixel_coordinates"]
        depth = sample["depth"]
        uncertainty_set = sample["uncertainty_set"]

        # 归一化
        pixel_coordinates = pixel_coordinates / ParameterRange.u_range[1]
        depth = (depth - ParameterRange.d_range[0]) / (
            ParameterRange.d_range[1] - ParameterRange.d_range[0]
        )

        uncertainty_set = (
            uncertainty_set
            - np.array(
                [
                    ParameterRange.f_x_range[0],
                    ParameterRange.f_x_range[0],
                    ParameterRange.f_y_range[0],
                    ParameterRange.f_y_range[0],
                    ParameterRange.c_x_range[0],
                    ParameterRange.c_x_range[0],
                    ParameterRange.c_y_range[0],
                    ParameterRange.c_y_range[0],
                    ParameterRange.rx_range[0],
                    ParameterRange.rx_range[0],
                    ParameterRange.ry_range[0],
                    ParameterRange.ry_range[0],
                    ParameterRange.rz_range[0],
                    ParameterRange.rz_range[0],
                    ParameterRange.tx_range[0],
                    ParameterRange.tx_range[0],
                    ParameterRange.ty_range[0],
                    ParameterRange.ty_range[0],
                    ParameterRange.tz_range[0],
                    ParameterRange.tz_range[0],
                ],
                dtype=np.float32,
            )
        ) / np.array(
            [
                ParameterRange.f_x_range[1] - ParameterRange.f_x_range[0],
                ParameterRange.f_x_range[1] - ParameterRange.f_x_range[0],
                ParameterRange.f_y_range[1] - ParameterRange.f_y_range[0],
                ParameterRange.f_y_range[1] - ParameterRange.f_y_range[0],
                ParameterRange.c_x_range[1] - ParameterRange.c_x_range[0],
                ParameterRange.c_x_range[1] - ParameterRange.c_x_range[0],
                ParameterRange.c_y_range[1] - ParameterRange.c_y_range[0],
                ParameterRange.c_y_range[1] - ParameterRange.c_y_range[0],
                ParameterRange.rx_range[1] - ParameterRange.rx_range[0],
                ParameterRange.rx_range[1] - ParameterRange.rx_range[0],
                ParameterRange.ry_range[1] - ParameterRange.ry_range[0],
                ParameterRange.ry_range[1] - ParameterRange.ry_range[0],
                ParameterRange.rz_range[1] - ParameterRange.rz_range[0],
                ParameterRange.rz_range[1] - ParameterRange.rz_range[0],
                ParameterRange.tx_range[1] - ParameterRange.tx_range[0],
                ParameterRange.tx_range[1] - ParameterRange.tx_range[0],
                ParameterRange.ty_range[1] - ParameterRange.ty_range[0],
                ParameterRange.ty_range[1] - ParameterRange.ty_range[0],
                ParameterRange.tz_range[1] - ParameterRange.tz_range[0],
                ParameterRange.tz_range[1] - ParameterRange.tz_range[0],
            ],
            dtype=np.float32,
        )

        # 将特征和标签组合为输入张量
        feature_tensor = torch.cat([pixel_coordinates, depth.unsqueeze(-1), uncertainty_set])

        sample["feature"] = feature_tensor

        return sample


class UncertaintyEllipsoidDataset(Dataset):
    """不确定性椭球数据集类"""

    def __init__(self, h5_path: Path, transform=FeatureCombiner):
        """
        初始化数据集

        Args:
            h5_path (Path): HDF5数据文件路径
            transform (callable, optional): 数据转换函数
        """
        self.h5_path = h5_path
        self.transform = transform

        # 打开HDF5文件
        # self.file = h5py.File(h5_path, "r")
        self.file = h5pickle.File(str(h5_path), "r")

        # 预先加载数据集的大小信息
        self.num_samples = self.file["world_coordinates"].shape[0]  # 样本数
        self.M_s = self.file["world_coordinates"].shape[1]  # MC采样数

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 读取数据
        world_coords = self.file["world_coordinates"][idx]  # shape: (M_s, 3)
        pixel_coords = self.file["pixel_coordinates"][idx]  # shape: (2,)
        depth = self.file["depths"][idx]  # shape: ()
        uncertainty_set = self.file["uncertainty_sets"][idx]  # shape: (20,)

        # 转换为torch张量
        sample = {
            "world_coordinates": torch.from_numpy(world_coords).float(),
            "pixel_coordinates": torch.from_numpy(pixel_coords).float(),
            "depth": torch.tensor(depth).float(),
            "uncertainty_set": torch.from_numpy(uncertainty_set).float(),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


app = typer.Typer()


class CameraParameters(BaseModel):
    """A data model to hold sampled intrinsic and extrinsic parameters."""

    # Intrinsic parameters
    f_x: float
    f_y: float
    c_x: float
    c_y: float

    # Extrinsic parameters
    rx: float
    ry: float
    rz: float
    tx: float
    ty: float
    tz: float


class UncertaintySet(BaseModel):
    # Intrinsic parameters (f_x, f_y)
    f_x: Tuple[float, float]
    f_y: Tuple[float, float]

    # Principal point (c_x, c_y)
    c_x: Tuple[float, float]
    c_y: Tuple[float, float]

    # Rotation (rx, ry, rz)
    rx: Tuple[float, float]
    ry: Tuple[float, float]
    rz: Tuple[float, float]

    # Translation (tx, ty, tz)
    tx: Tuple[float, float]
    ty: Tuple[float, float]
    tz: Tuple[float, float]


# Define the uncertainty range for each parameter (20-tuple)
class ParameterRange:
    u_range = (0.0, 480.0)
    v_range = (0.0, 640.0)
    d_range = (0.2, 0.7)

    f_x_range = (595.0, 615.0)
    f_y_range = (595.0, 615.0)

    # 主点范围 (c_x, c_y):
    c_x_range = (290.0, 330.0)
    c_y_range = (230.0, 270.0)

    # 旋转矩阵 (rx, ry, rz):
    rx_range = (0.75, 1.75)
    ry_range = (-1.75, -0.75)
    rz_range = (0.75, 1.75)

    # 平移向量 (tx, ty, tz):
    tx_range = (-0.35, 0.25)
    ty_range = (-0.35, 0.25)
    tz_range = (-0.25, -0.05)


def sample_uncertainty_set() -> UncertaintySet:
    # Sample a random interval (subset) for each parameter range

    # Intrinsic parameters (f_x, f_y)
    f_x_min = np.random.uniform(ParameterRange.f_x_range[0], ParameterRange.f_x_range[1] - 1)
    f_x_max = np.random.uniform(f_x_min + 1, ParameterRange.f_x_range[1])

    f_y_min = np.random.uniform(ParameterRange.f_y_range[0], ParameterRange.f_y_range[1] - 1)
    f_y_max = np.random.uniform(f_y_min + 1, ParameterRange.f_y_range[1])

    # Principal point (c_x, c_y)
    c_x_min = np.random.uniform(ParameterRange.c_x_range[0], ParameterRange.c_x_range[1] - 1)
    c_x_max = np.random.uniform(c_x_min + 1, ParameterRange.c_x_range[1])

    c_y_min = np.random.uniform(ParameterRange.c_y_range[0], ParameterRange.c_y_range[1] - 1)
    c_y_max = np.random.uniform(c_y_min + 1, ParameterRange.c_y_range[1])

    # Rotation (rx, ry, rz)
    rx_min = np.random.uniform(ParameterRange.rx_range[0], ParameterRange.rx_range[1] - 0.1)
    rx_max = np.random.uniform(rx_min + 0.1, ParameterRange.rx_range[1])

    ry_min = np.random.uniform(ParameterRange.ry_range[0], ParameterRange.ry_range[1] - 0.1)
    ry_max = np.random.uniform(ry_min + 0.1, ParameterRange.ry_range[1])

    rz_min = np.random.uniform(ParameterRange.rz_range[0], ParameterRange.rz_range[1] - 0.1)
    rz_max = np.random.uniform(rz_min + 0.1, ParameterRange.rz_range[1])

    # Translation (tx, ty, tz)
    tx_min = np.random.uniform(ParameterRange.tx_range[0], ParameterRange.tx_range[1] - 0.05)
    tx_max = np.random.uniform(tx_min + 0.05, ParameterRange.tx_range[1])

    ty_min = np.random.uniform(ParameterRange.ty_range[0], ParameterRange.ty_range[1] - 0.05)
    ty_max = np.random.uniform(ty_min + 0.05, ParameterRange.ty_range[1])

    tz_min = np.random.uniform(ParameterRange.tz_range[0], ParameterRange.tz_range[1] - 0.05)
    tz_max = np.random.uniform(tz_min + 0.05, ParameterRange.tz_range[1])

    # Create an instance of the UncertaintySet model
    uncertainty_set = UncertaintySet(
        f_x=(f_x_min, f_x_max),
        f_y=(f_y_min, f_y_max),
        c_x=(c_x_min, c_x_max),
        c_y=(c_y_min, c_y_max),
        rx=(rx_min, rx_max),
        ry=(ry_min, ry_max),
        rz=(rz_min, rz_max),
        tx=(tx_min, tx_max),
        ty=(ty_min, ty_max),
        tz=(tz_min, tz_max),
    )

    return uncertainty_set


def sample_camera_parameters(uncertainty_set: UncertaintySet) -> CameraParameters:
    """Samples a set of intrinsic and extrinsic parameters from an UncertaintySet."""

    # Sample each parameter within its interval
    f_x = np.random.uniform(*uncertainty_set.f_x)
    f_y = np.random.uniform(*uncertainty_set.f_y)

    c_x = np.random.uniform(*uncertainty_set.c_x)
    c_y = np.random.uniform(*uncertainty_set.c_y)

    rx = np.random.uniform(*uncertainty_set.rx)
    ry = np.random.uniform(*uncertainty_set.ry)
    rz = np.random.uniform(*uncertainty_set.rz)

    tx = np.random.uniform(*uncertainty_set.tx)
    ty = np.random.uniform(*uncertainty_set.ty)
    tz = np.random.uniform(*uncertainty_set.tz)

    # Create and return a CameraParameters object
    return CameraParameters(
        f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y, rx=rx, ry=ry, rz=rz, tx=tx, ty=ty, tz=tz
    )


def compute_world_coordinates(
    u: float, v: float, d: float, camera_params: CameraParameters
) -> Tuple[float, float, float]:
    """
    Compute the world coordinates (X, Y, Z) given pixel coordinates (u, v), depth d,
    and camera intrinsic & extrinsic parameters.

    Args:
        u (float): Pixel x-coordinate.
        v (float): Pixel y-coordinate.
        d (float): Depth value.
        camera_params (CameraParameters): Camera intrinsic and extrinsic parameters.

    Returns:
        Tuple[float, float, float]: The computed world coordinates (X, Y, Z).
    """

    # Compute camera coordinates
    x_c = (u - camera_params.c_x) * d / camera_params.f_x
    y_c = (v - camera_params.c_y) * d / camera_params.f_y
    z_c = d

    # Camera coordinate vector
    P_c = np.array([x_c, y_c, z_c])

    # Compute the rotation matrix from the rotation vector (rx, ry, rz)
    rotation_vector = np.array([camera_params.rx, camera_params.ry, camera_params.rz])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()

    # Compute the translation vector
    translation_vector = np.array([camera_params.tx, camera_params.ty, camera_params.tz])

    # Transform camera coordinates to world coordinates
    world_coords = np.dot(rotation_matrix, P_c) + translation_vector

    return tuple(world_coords)


def process_batch(batch_idx: int, batch_size: int, M_s: int) -> tuple:
    """处理一个批次的数据"""
    world_coords_batch = np.zeros((batch_size, M_s, 3), dtype=np.float32)
    pixel_coords_batch = np.zeros((batch_size, 2), dtype=np.float32)
    depths_batch = np.zeros(batch_size, dtype=np.float32)
    uncertainty_sets_batch = np.zeros((batch_size, 20), dtype=np.float32)

    for i in range(batch_size):
        # 采样参数不确定性集
        uncertainty_set = sample_uncertainty_set()

        # 随机采样像素坐标
        u = np.random.uniform(*ParameterRange.u_range)
        v = np.random.uniform(*ParameterRange.v_range)

        # 采样深度
        d = np.random.uniform(*ParameterRange.d_range)

        # 生成蒙特卡洛样本
        world_coords = np.array(
            [
                compute_world_coordinates(u, v, d, sample_camera_parameters(uncertainty_set))
                for _ in range(M_s)
            ]
        )

        # 存储数据
        world_coords_batch[i] = world_coords
        pixel_coords_batch[i] = [u, v]
        depths_batch[i] = d

        # 展平uncertainty_set
        flattened_uncertainty = np.array(
            [value for interval in uncertainty_set.dict().values() for value in interval]
        )
        uncertainty_sets_batch[i] = flattened_uncertainty

    return (
        batch_idx,
        world_coords_batch,
        pixel_coords_batch,
        depths_batch,
        uncertainty_sets_batch,
    )


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.h5",
    num_samples: int = int(1e6),
    M_s: int = 100,
    batch_size: int = 1000,  # 每个批次的样本数
    num_processes: int = mp.cpu_count(),  # 使用CPU核心数
):
    """
    Generate a dataset by sampling intrinsic and extrinsic camera parameters,
    computing world coordinates, and storing the results in HDF5 format.
    """
    logger.info("开始处理数据集...")
    logger.info(f"使用 {num_processes} 个进程进行并行处理")

    # 计算批次数
    num_batches = (num_samples + batch_size - 1) // batch_size

    # 创建HDF5文件和数据集
    with h5py.File(output_path, "w") as f:
        # Create datasets
        world_coords_dset = f.create_dataset(
            "world_coordinates",
            shape=(num_samples, M_s, 3),
            dtype=np.float32,
            chunks=True,  # 启用分块存储
            compression="gzip",
            compression_opts=4,
        )
        pixel_coords_dset = f.create_dataset(
            "pixel_coordinates",
            shape=(num_samples, 2),
            dtype=np.float32,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
        depths_dset = f.create_dataset(
            "depths",
            shape=(num_samples,),
            dtype=np.float32,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
        uncertainty_set_dset = f.create_dataset(
            "uncertainty_sets",
            shape=(num_samples, 20),
            dtype=np.float32,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )

        # 创建进程池
        with mp.Pool(num_processes) as pool:
            # 准备并行处理的参数
            process_func = partial(process_batch, batch_size=batch_size, M_s=M_s)

            # 使用tqdm显示进度
            for result in tqdm(
                pool.imap_unordered(process_func, range(num_batches)),
                total=num_batches,
                desc="生成数据集",
            ):
                batch_idx, world_coords, pixel_coords, depths, uncertainty_sets = result

                # 计算实际的批次大小（最后一个批次可能较小）
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                actual_batch_size = end_idx - start_idx

                # 写入数据
                world_coords_dset[start_idx:end_idx] = world_coords[:actual_batch_size]
                pixel_coords_dset[start_idx:end_idx] = pixel_coords[:actual_batch_size]
                depths_dset[start_idx:end_idx] = depths[:actual_batch_size]
                uncertainty_set_dset[start_idx:end_idx] = uncertainty_sets[:actual_batch_size]

                # 定期刷新缓存
                f.flush()

    logger.success("数据集处理完成！")


if __name__ == "__main__":
    # 设置随机种子确保可重复性
    np.random.seed(42)
    app()
