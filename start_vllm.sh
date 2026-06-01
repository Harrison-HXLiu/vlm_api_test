#!/bin/bash
set -e

MODEL_PATH=${MODEL_PATH:-/models/Qwen2.5-VL-7B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen2.5-vl-7b}
API_KEY=${API_KEY:-your_secret_token}
PORT=${PORT:-8000}

python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_PATH} \
  --tokenizer ${MODEL_PATH} \
  --tokenizer-mode auto \
  --host 0.0.0.0 \
  --port ${PORT} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --trust-remote-code \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.85 \
  --api-key ${API_KEY}