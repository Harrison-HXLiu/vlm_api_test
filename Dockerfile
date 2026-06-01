FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --timeout 1000

RUN pip install vllm==0.8.5.post1 \
    transformers==4.51.3 \
    qwen-vl-utils \
    pillow \
    opencv-python \
    fastapi \
    uvicorn \
    python-multipart \
    openai \
    python-dotenv \
    --timeout 1000

COPY start_vllm.sh /app/start_vllm.sh

RUN chmod +x /app/start_vllm.sh

EXPOSE 8000

CMD ["/app/start_vllm.sh"]