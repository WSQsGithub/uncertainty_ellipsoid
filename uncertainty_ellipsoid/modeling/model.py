from pathlib import Path
from re import L

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class EllipsoidNet(nn.Module):
    def __init__(self):
        super(EllipsoidNet, self).__init__()
        # 输入层 23 个神经元
        self.fc1 = nn.Linear(23, 64)
        # 三个隐藏层，每个层有 64 个神经元
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        # 输出层 9 个神经元 (3 个用于中心，6 个用于精度矩阵下三角元素)
        self.fc4 = nn.Linear(64, 9)

    def forward(self, x):
        # 前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        outputs = self.fc4(x)

        # 将输出分解为中心和精度矩阵的下三角元素
        c_x, c_y, c_z, l_11, l_21, l_31, l_22, l_32, l_33 = torch.chunk(outputs, 9, dim=-1)

        # 将中心 c_x, c_y, c_z 合并为一个 3D 向量
        center = torch.stack((c_x, c_y, c_z), dim=-1).squeeze(1)

        # 使精度矩阵的下三角元素为正，使用 softplus 保证正数
        l_11 = F.softplus(l_11)  # 保证 l_11 为正
        l_22 = F.softplus(l_22)  # 保证 l_22 为正
        l_33 = F.softplus(l_33)  # 保证 l_33 为正

        # 由于是下三角矩阵，我们不需要对 l_21, l_31, l_32 进行归一化，只保证其为正
        l_21 = F.softplus(l_21)  # 保证 l_21 为正
        l_31 = F.softplus(l_31)  # 保证 l_31 为正
        l_32 = F.softplus(l_32)  # 保证 l_32 为正

        # 构造 3x3 下三角矩阵 L（支持批量计算）
        batch_size = x.size(0)
        L = torch.zeros((batch_size, 3, 3), device=x.device)

        # 填充 L 矩阵的下三角部分
        L[:, 0, 0] = l_11.squeeze(-1)  # 确保形状匹配
        L[:, 1, 0] = l_21.squeeze(-1)
        L[:, 2, 0] = l_31.squeeze(-1)
        L[:, 1, 1] = l_22.squeeze(-1)
        L[:, 2, 1] = l_32.squeeze(-1)
        L[:, 2, 2] = l_33.squeeze(-1)

        # 返回中心和 L 矩阵
        return center, L
    

class EllipsoidCenterNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 输入层 23 个神经元
        self.fc1 = nn.Linear(23, 64)
        # 三个隐藏层，每个层有 64 个神经元
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        # 输出层 3 个神经元 (3 个中心坐标)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        center = self.fc4(x)
        return center
    
class EllipsoidShapeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 输入层 23 + 3 个神经元
        self.fc1 = nn.Linear(26, 64)
        # 三个隐藏层，每个层有 64 个神经元
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        # 输出层 6 个神经元 (精度矩阵下三角元素)
        self.fc4 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        outputs = self.fc4(x)

        # 将输出分解为精度矩阵的下三角元素
        l_11, l_21, l_31, l_22, l_32, l_33 = torch.chunk(outputs, 6, dim=-1)

        # 使精度矩阵的下三角元素为正，使用 softplus 保证正数
        l_11 = F.softplus(l_11)
        l_22 = F.softplus(l_22)
        l_33 = F.softplus(l_33)
        l_21 = F.softplus(l_21)
        l_31 = F.softplus(l_31)
        l_32 = F.softplus(l_32)
        
        # 构造 3x3 下三角矩阵 L（支持批量计算）
        batch_size = x.size(0)
        L = torch.zeros((batch_size, 3, 3), device=x.device)

        # 填充 L 矩阵的下三角部分
        L[:, 0, 0] = l_11.squeeze(-1)  # 确保形状匹配
        L[:, 1, 0] = l_21.squeeze(-1)
        L[:, 2, 0] = l_31.squeeze(-1)
        L[:, 1, 1] = l_22.squeeze(-1)
        L[:, 2, 1] = l_32.squeeze(-1)
        L[:, 2, 2] = l_33.squeeze(-1)

        # 返回中心和 L 矩阵
        return L



def safe_load_model(model_path: Path, device: torch.device, task: str) -> EllipsoidNet:
    """Safely load model, automatically handle missing weights"""
    if task == "end2end":
        model = EllipsoidNet()
    elif task == "train_center":
        model = EllipsoidCenterNet()
    elif task == "train_shape":
        model = EllipsoidShapeNet()

    try:
        # Try to load pretrained weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded model weights: {model_path}")
    except FileNotFoundError:
        logger.warning("Pretrained weights not found, using random initialization")
        # Keep default random initialization
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # 创建模型实例
    # model = EllipsoidNet()
    model = EllipsoidCenterNet()

    # 打印模型架构
    print(model)

    # 假设我们有一个输入样本，形状为 (batch_size, 23)
    sample_input = torch.randn(5, 23)  # 示例输入

    # 获取输出 center 和 L
    # center, L = model(sample_input)
    center = model(sample_input)
    print("center shape:", center.shape)  # (batch_size, 3)
    print("Center:", center)
    # print("L shape:", L.shape)  # (batch_size, 3, 3)
    # print("L:", L)
