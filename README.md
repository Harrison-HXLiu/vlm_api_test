# 测试环境
torch：2.6.0+cu124
Python 3.10.20
CUDA Version: 12.4
vllm：0.8.5.post1

# 安装依赖
```
pip install -U transformers accelerate qwen-vl-utils modelscope
pip install -U pillow opencv-python-headless pandas requests
pip install -U fastapi uvicorn python-multipart
pip install -U torchvision
```

# 目录结构
```
mkdir -p /root/vlm_api
cd /root/vlm_api

mkdir -p models
mkdir -p test_images
mkdir -p outputs
mkdir -p logs
```
/root/vlm_api/
  models/
  test_images/
  outputs/
  logs/
  run_vlm_once.py
  server_vlm.py

# 下载模型
```
cd /root/vlm_api

modelscope download \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --local_dir /root/vlm_api/models/Qwen2.5-VL-7B-Instruct
```

# 运行测试
```
conda activate vlm
cd /root/vlm_api

python face_benchmark.py \
  --model-path /root/vlm_api/models/Qwen2.5-VL-7B-Instruct \
  --tasks all \
  --warmup 3 \
  --repeat 1 \
  --max-new-tokens 48 \
  --attn-implementation sdpa
```

# 开启接口
python -m vllm.entrypoints.openai.api_server \
  --model /root/vlm_api/models/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2.5-vl-7b \
  --trust-remote-code \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.85 \
  --api-key your api

# 调用示例
from openai import OpenAI

client = OpenAI(
    api_key="your api",
    base_url="http://42.193.241.119:35457/v1"
)

resp = client.chat.completions.create(
    model="qwen2.5-vl-7b",
    messages=[
        {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    max_tokens=100
)

print(resp.choices[0].message.content)