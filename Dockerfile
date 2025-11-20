# ============================================================
# DeepSeek-OCR + vLLM (nightly) with Pre-Baked Model
# ============================================================
# Use vLLM nightly image from Docker Hub (includes PyTorch + CUDA + vLLM)
FROM vllm/vllm-openai:nightly
# Set Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/.cache/huggingface
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
# Set the Working Directory
WORKDIR /workspace
# ------------------------------------------------------------
# 1️⃣ Install system dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
        wget curl git build-essential ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# ------------------------------------------------------------
# 2️⃣ Install Python dependencies from requirements.txt
# ------------------------------------------------------------
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt
# ------------------------------------------------------------
# 3️⃣ Download DeepSeek-OCR model into the image
# ------------------------------------------------------------
RUN python3 - << 'EOF'
from huggingface_hub import snapshot_download
import os
print("📥 Downloading DeepSeek-OCR model into /workspace/model ...")
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-OCR",
    local_dir="/workspace/model",
    local_dir_use_symlinks=False,
)
print("✅ Model downloaded successfully!")
EOF
# ------------------------------------------------------------
# 4️⃣ Copy application code
# ------------------------------------------------------------
COPY . /workspace/
# ------------------------------------------------------------
# 5️⃣ Final setup steps
# ------------------------------------------------------------
RUN chmod +x /workspace/entrypoint.sh
EXPOSE 7860 8888
ENTRYPOINT ["/workspace/entrypoint.sh"]