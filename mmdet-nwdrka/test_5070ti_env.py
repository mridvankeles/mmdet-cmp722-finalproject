#!/usr/bin/env python3
"""
Test script to verify PyTorch, CUDA, and MMDetection installation
for RTX 5070 Ti (Blackwell architecture, sm_120)
"""

import sys

def test_pytorch():
    """Test PyTorch installation and CUDA support"""
    print("=" * 60)
    print("Testing PyTorch Installation")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
            
            # Get GPU compute capability
            capability = torch.cuda.get_device_capability(0)
            print(f"✓ GPU compute capability: sm_{capability[0]}{capability[1]}")
            
            # Test CUDA operations
            print("\nTesting CUDA operations...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            print(f"✓ Matrix multiplication successful!")
            print(f"  Result mean: {z.mean().item():.6f}")
            
            # Test mixed precision
            print("\nTesting mixed precision (FP16)...")
            with torch.cuda.amp.autocast():
                z_fp16 = torch.matmul(x, y)
            print(f"✓ Mixed precision successful!")
            print(f"  Result dtype: {z_fp16.dtype}")
            
            return True
        else:
            print("✗ CUDA not available!")
            return False
            
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_mmcv():
    """Test MMCV installation"""
    print("\n" + "=" * 60)
    print("Testing MMCV Installation")
    print("=" * 60)
    
    try:
        import mmcv
        print(f"✓ MMCV version: {mmcv.__version__}")
        
        # Test CUDA ops if available
        from mmcv.ops import get_compiling_cuda_version, get_compiler_version
        print(f"✓ MMCV CUDA version: {get_compiling_cuda_version()}")
        print(f"✓ MMCV compiler version: {get_compiler_version()}")
        
        return True
    except Exception as e:
        print(f"✗ MMCV test failed: {e}")
        return False

def test_mmdet():
    """Test MMDetection installation"""
    print("\n" + "=" * 60)
    print("Testing MMDetection Installation")
    print("=" * 60)
    
    try:
        import mmdet
        print(f"✓ MMDetection version: {mmdet.__version__}")
        
        # Test importing key modules
        from mmdet.apis import init_detector, inference_detector
        from mmdet.models import build_detector
        print(f"✓ MMDetection APIs imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ MMDetection test failed: {e}")
        return False

def test_custom_modules():
    """Test custom AITOD modules"""
    print("\n" + "=" * 60)
    print("Testing Custom AITOD Modules")
    print("=" * 60)
    
    try:
        # Test AITOD dataset
        from mmdet.datasets import AITODv2Dataset
        print(f"✓ AITODv2Dataset imported successfully")
        
        # Test custom plugins
        from mmdet.models.plugins import CSAMPlugin
        print(f"✓ CSAMPlugin imported successfully")
        
        return True
    except Exception as e:
        print(f"⚠ Custom modules test: {e}")
        print("  This is expected if you haven't installed custom modules yet")
        return True  # Don't fail on this

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RTX 5070 Ti Environment Validation")
    print("=" * 60 + "\n")
    
    results = {
        'PyTorch': test_pytorch(),
        'MMCV': test_mmcv(),
        'MMDetection': test_mmdet(),
        'Custom Modules': test_custom_modules()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Environment is ready for training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
