# ==================================================================
# LTX-2 RunPod Serverless Dockerfile  
# CUDA 12.8.0 | Ubuntu 22.04 | Python 3.11
# Fixed: Using uv for fast, reliable dependency resolution
# ==================================================================

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# ==================================================================
# Environment Configuration
# ==================================================================
ENV CUDA_HOME=/usr/local/cuda

ENV PATH="/root/.cargo/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    HUGGINGFACE_HUB_CACHE=/workspace/hf_cache \
    MODEL_CACHE_DIR=/workspace/models \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9" \
    OMP_NUM_THREADS=4

# ==================================================================
# System Dependencies
# FIX: Removed python3-pip (using uv instead), kept python3.10-venv for compatibility
# ==================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    curl \
    ca-certificates \
    git \
    wget \
    ffmpeg \
    libavcodec58 \
    libavformat58 \
    libavdevice58 \
    libavutil56 \
    libswscale5 \
    libswresample3 \
    libavfilter7 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get clean

# Set Python 3.11 as default (removed pip3 alternative since we use uv)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ==================================================================
# Install uv (Fast Python Package Manager)
# ==================================================================
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ==================================================================
# Working Directory
# ==================================================================
WORKDIR /workspace

# ==================================================================
# Clone LTX-2 Repository
# ==================================================================
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /workspace/ltx2-repo

# ==================================================================
# Install Python Dependencies with uv
# ==================================================================
COPY requirements.txt /workspace/requirements.txt

# Install torch ecosystem (CUDA 12.8) using uv
ENV UV_HTTP_TIMEOUT=10000000 UV_HTTP_RETRIES=100
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    triton==3.5.1 \
    xformers==0.0.33

# Install remaining dependencies (uv auto-resolves conflicts)
RUN uv pip install --system --index-strategy unsafe-best-match -r /workspace/requirements.txt

# Verify torch installation (CPU check only)
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Runtime: {torch.version.cuda}'); \
    print('Build successful - CUDA will be available at runtime')"

# ==================================================================
# Install LTX-2 Local Packages
# ==================================================================
RUN uv pip install --system --no-deps -e /workspace/ltx2-repo/packages/ltx-core && \
    uv pip install --system --no-deps -e /workspace/ltx2-repo/packages/ltx-pipelines && \
    python -c "import ltx_core; import ltx_pipelines; print('✓ LTX packages installed')"

# ==================================================================
# Application Code
# ==================================================================
COPY handler.py /workspace/handler.py
COPY inference.py /workspace/inference.py

# ==================================================================
# Cache Directories & Permissions
# ==================================================================
RUN mkdir -p /workspace/models /workspace/hf_cache /workspace/temp /workspace/outputs && \
    chmod -R 777 /workspace

# ==================================================================
# Health Check for RunPod
# ==================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import torch; import ltx_core; import ltx_pipelines; assert torch.cuda.is_available(), 'CUDA not available!'; print('Healthy')" || exit 1

# ==================================================================
# Entrypoint
# ==================================================================
CMD ["python", "/workspace/handler.py"]