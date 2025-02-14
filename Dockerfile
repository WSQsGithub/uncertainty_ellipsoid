# 基础镜像选择PyTorch官方镜像（包含CUDA 11.8和cuDNN 8）
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app
ENV PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    hdf5-tools \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY Makefile .
COPY pyproject.toml .
COPY setup.cfg .

# 安装Python依赖（使用清华镜像加速）
RUN pip install --upgrade pip && \
    pip install torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 复制项目代码
COPY uncertainty_ellipsoid/ ./uncertainty_ellipsoid

# 配置非root用户
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 设置默认命令（可根据需要覆盖）
CMD ["make", "api"] 

# 构建镜像
# docker build -t uncertainty-ellipsoid .

# 运行训练（使用GPU）
# docker run --gpus all -v $(pwd)/data:/app/data uncertainty-ellipsoid make train

# 运行API服务
# docker run -p 8000:8000 -v $(pwd)/models:/app/models uncertainty-ellipsoid