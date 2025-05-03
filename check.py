# Chạy trong python hoặc lưu thành file .py rồi chạy
import torch
if torch.cuda.is_available():
    print("--- SUCCESS: CUDA is available! ---")
    print("Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Version PyTorch built with:", torch.version.cuda)
else:
    print("--- ERROR: CUDA is NOT available. ---")
    # In thêm thông tin nếu có thể
    try:
        torch._C._cuda_init() 
    except RuntimeError as e:
        print("Error during manual CUDA init:", e)