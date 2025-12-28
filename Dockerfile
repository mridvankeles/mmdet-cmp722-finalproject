# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# ----------------------------------------------------------------------
# System dependencies
# ----------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git curl wget nano openssh-client \
    build-essential gcc-9 g++-9 make cmake pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libjpeg-dev libpng-dev ffmpeg \
 && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-9 \
    CXX=/usr/bin/g++-9

# ----------------------------------------------------------------------
# Miniforge (Conda) setup
# ----------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
 && rm -f /tmp/miniforge.sh \
 && ${CONDA_DIR}/bin/conda config --system --add channels conda-forge \
 && ${CONDA_DIR}/bin/conda config --system --set channel_priority strict

ENV PATH=${CONDA_DIR}/bin:${PATH}
SHELL ["/bin/bash", "-c"]

# ----------------------------------------------------------------------
# Create env (Python 3.8) and install PyTorch/cu117 via PIP (avoids MKL issue)
# ----------------------------------------------------------------------
ARG PYTHON_VERSION=3.8
RUN conda create -y -n mmdet python=${PYTHON_VERSION} numpy ninja && conda clean -ya
ENV CONDA_DEFAULT_ENV=mmdet
ENV PATH=${CONDA_DIR}/envs/${CONDA_DEFAULT_ENV}/bin:${CONDA_DIR}/bin:${PATH}

# Keep pip/setuptools compatible with Python 3.8; then install torch/cu117 wheels
RUN source activate mmdet \
 && python -m pip install --no-cache-dir "pip<25" "setuptools<70" "wheel<0.41" \
 && pip install --no-cache-dir \
      torch==1.13.1+cu117 \
      torchvision==0.14.1+cu117 \
      torchaudio==0.13.1 \
      --extra-index-url https://download.pytorch.org/whl/cu117

# ----------------------------------------------------------------------
# Install mmcv-full (prebuilt wheel for torch1.13/cu117) and mmdetection 2.24.1
# ----------------------------------------------------------------------
ENV MMCV_WITH_OPS=1 FORCE_CUDA=1
RUN source activate mmdet \
 && pip install --no-cache-dir -U openmim \
 && mim install "mmcv-full==1.7.1" \
 && pip install --no-cache-dir "mmdet==2.24.1"

# ----------------------------------------------------------------------
# Copy project
# ----------------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace

# ----------------------------------------------------------------------
# Project requirements + AITOD pycocotools + editable install
# ----------------------------------------------------------------------
RUN source activate mmdet \
 && pip install --no-cache-dir "cython<3" \
 && pip install --no-cache-dir -r mmdet-nwdrka/requirements/build.txt \
 && pip install --no-cache-dir -r mmdet-nwdrka/requirements/runtime.txt \
 && pip install --no-cache-dir -r mmdet-nwdrka/requirements/optional.txt \
 && pip install --no-cache-dir "yapf==0.40.1" \
 && cd cocoapi-aitod-master/aitodpycocotools && pip install --no-cache-dir -v . \
 && cd /workspace/mmdet-nwdrka && pip install --no-cache-dir -v -e .

# ----------------------------------------------------------------------
# Default to GPUs 0,1 (can be overridden at runtime)
# ----------------------------------------------------------------------
ENV CUDA_VISIBLE_DEVICES=0,1 \
    NVIDIA_VISIBLE_DEVICES=0,1

WORKDIR /workspace/mmdet-nwdrka
CMD ["bash"]

