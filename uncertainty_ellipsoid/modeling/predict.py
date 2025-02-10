from pathlib import Path
import torch
import pandas as pd
import typer
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from uncertainty_ellipsoid.config import MODELS_DIR, PROCESSED_DATA_DIR
from uncertainty_ellipsoid.dataset import FeatureCombiner, get_dataloader
from uncertainty_ellipsoid.modeling.model import safe_load_model

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.h5",
    model_path: Path = MODELS_DIR / "ellipsoid_net.pth",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    执行模型预测，自动处理模型初始化
    
    Args:
        features_path: 预处理后的特征数据路径
        model_path: 模型权重文件路径
        predictions_path: 预测结果输出路径
        batch_size: 预测批次大小
        device: 计算设备 (cuda/cpu)
    """
    # 初始化设备
    device = torch.device(device)
    logger.info(f"使用计算设备: {device}")
    
    # 初始化数据流水线
    logger.info("Preparing dataloader...")
    dataloader = get_dataloader(
        h5_path=features_path,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
        transform=FeatureCombiner()
    )
    
    # 初始化模型
    logger.info(f"Safe loading model from {model_path} to {device}")
    model = safe_load_model(model_path, device)
    
    # 执行预测
    logger.info("Starting prediction...")
    predictions = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset), desc="预测进度") as pbar:
        for batch in dataloader:
            inputs = batch["input"].to(device, non_blocking=True)
            
            # 模型推理
            centers, L_elements = model(inputs)
            
            # 转换到CPU
            batch_results = {
                "center_x": centers[:, 0].cpu().numpy(),
                "center_y": centers[:, 1].cpu().numpy(),
                "center_z": centers[:, 2].cpu().numpy(),
                "L11": L_elements[:, 0, 0].cpu().numpy(),
                "L21": L_elements[:, 1, 0].cpu().numpy(),
                "L31": L_elements[:, 2, 0].cpu().numpy(),
                "L22": L_elements[:, 1, 1].cpu().numpy(),
                "L32": L_elements[:, 2, 1].cpu().numpy(),
                "L33": L_elements[:, 2, 2].cpu().numpy(),
            }
            predictions.append(pd.DataFrame(batch_results))
            pbar.update(len(inputs))
    
    # 保存结果
    full_df = pd.concat(predictions, ignore_index=True)
    full_df.to_csv(predictions_path, index=False)
    logger.success(f"预测结果已保存至: {predictions_path}")

if __name__ == "__main__":
    app()
