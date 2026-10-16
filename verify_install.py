import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. PyTorch is running on CPU.")
# Test LoRA package loading
try:
    import peft
    print("PEFT (LoRA) successfully installed!")
except ImportError:
    try:
        import lora_pytorch
        print("lora-pytorch successfully installed!")
    except ImportError:
        print("LoRA package not found.")
