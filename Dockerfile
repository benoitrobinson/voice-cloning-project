# Multi-stage build for optimized image size
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data/voices /app/data/recordings /app/logs

# Download models (can be skipped if models are mounted)
ARG DOWNLOAD_MODELS=false
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
    python3 scripts/download_models.py; \
    fi

# Create non-root user
RUN useradd -m -u 1000 voiceuser && \
    chown -R voiceuser:voiceuser /app

USER voiceuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage with additional tools
FROM base AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter for notebooks
RUN pip3 install jupyter jupyterlab

# Expose Jupyter port
EXPOSE 8888

USER voiceuser

# Production stage with security hardening
FROM base AS production

# Remove unnecessary packages and files
USER root
RUN apt-get purge -y \
    git \
    wget \
    curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && find /app -name "*.pyc" -delete \
    && find /app -name "__pycache__" -type d -exec rm -rf {} +

# Set production environment
ENV ENVIRONMENT=production

USER voiceuser

# Use gunicorn for production
RUN pip3 install gunicorn

CMD ["gunicorn", "api.app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
