import torch
print(torch.cuda.is_available())
print("PyTorch 版本:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("cuDNN 版本:", torch.backends.cudnn.version())
print("cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 名稱:", torch.cuda.get_device_name(0))
