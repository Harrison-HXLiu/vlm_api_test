# 测试环境
- python 3.10
- torch：2.5.1+cu121
- Cuda: 12.1
- vllm

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