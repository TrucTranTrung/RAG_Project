from huggingface_hub import snapshot_download
import os

# Tên model trên Hugging Face Hub
model_id = "Systran/faster-whisper-medium"

# Thư mục cache mặc định của Hugging Face
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print(f"Downloading model '{model_id}' to cache directory '{cache_dir}'...")

# Tải toàn bộ các file của model
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    # ignore_patterns=["*.pt", "*.safetensors"], 
)

print("Model downloaded successfully.")