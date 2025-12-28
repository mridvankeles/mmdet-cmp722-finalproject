
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.matmul(x, y)
        print("Matrix multiplication successful!")
        print(z.mean())
    except Exception as e:
        print("Caught exception:")
        print(e)
else:
    print("CUDA not available")
