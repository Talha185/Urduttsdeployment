# Base image with CUDA 12.6 and Python 3.10 (adjust if you're using a specific PyTorch base)
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# Set environment variables to avoid prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 and pip3 to standard names if not already
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app
RUN pip install --break-system-packages torch==2.6.0+cu126 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --break-system-packages -r requirements.txt


# Expose the port FastAPI runs on
EXPOSE 8001

# Run the FastAPI application
CMD ["python3", "apper.py"]
