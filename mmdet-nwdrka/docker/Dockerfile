      
# 1. DEFINE ARGUMENTS FOR A MODERN, PYTHON-COMPATIBLE STACK
# CHANGED: Updated to a PyTorch version with a valid tag on Docker Hub
ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

# 2. USE A PYTORCH BASE IMAGE THAT EXISTS ON DOCKER HUB
# The tag pytorch:1.10.1-...-devel did not exist, so we use 1.11.0
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 2. USE A PYTORCH BASE IMAGE THAT EXISTS ON DOCKER HUB
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 3. SET ENVIRONMENT VARIABLES FOR BUILDING
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# Prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
# 4. INSTALL SYSTEM DEPENDENCIES
# Using --fix-missing in case of transient repository issues
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    ninja-build \
    libglib2.0-0 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. INSTALL MMCV USING MIM (THE RECOMMENDED WAY)
# Upgrading pip and installing mim for robust dependency handling
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir openmim
# Let mim install the version of mmcv-full compatible with PyTorch 1.11.0
# CHANGED: Updated mmcv-full to a version compatible with PyTorch 1.11.0
RUN mim install mmcv-full==1.6.0

# 6. INSTALL MMDETECTION
# Clone a specific, stable version of MMDetection for a reproducible build
RUN git clone -b v2.25.0 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection

# Set FORCE_CUDA to ensure CUDA extensions are built
ENV FORCE_CUDA="1"
# Install MMDetection in editable mode
# This also installs its dependencies from requirements.txt
RUN pip install --no-cache-dir -e .

# 7. VERIFY THE INSTALLATION
# This is a good practice to ensure the build was successful
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); import mmcv; print(f'MMCV version: {mmcv.__version__}'); from mmdet.apis import init_detector; print('MMDetection installation appears successful.')"

    
