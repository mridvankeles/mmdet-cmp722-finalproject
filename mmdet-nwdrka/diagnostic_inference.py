"""
Diagnostic script to check if the model is actually running on GPU
and if it's producing any outputs at all
"""
import torch
from mmdet.apis import init_detector, inference_detector
import numpy as np

CONFIG_FILE = 'workdir/exp2_super_attn_faster_rcnn/aitod_super_faster_r50_csam_nwdrka_1x.py'
CHECKPOINT_FILE = 'workdir/exp2_super_attn_faster_rcnn/epoch_30.pth'
IMAGE_FILE = 'data/aitod_dataset/aitodv2/trainval/P2754__1.0__1800___1200.png'

print("=" * 60)
print("CUDA Diagnostic")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

print("\n" + "=" * 60)
print("Loading model...")
print("=" * 60)

# Try loading on CPU first
print("\nTesting on CPU...")
model_cpu = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')
print("Model loaded on CPU successfully!")

# Disable TTA
model_cpu.cfg.data.test.pipeline[1]['img_scale'] = (800, 800)

# Run inference on CPU
print("Running inference on CPU...")
result_cpu = inference_detector(model_cpu, IMAGE_FILE)

print(f"\nCPU Result type: {type(result_cpu)}")
if isinstance(result_cpu, list):
    total_cpu = sum(len(dets) for dets in result_cpu)
    print(f"Total detections on CPU: {total_cpu}")
    for i, dets in enumerate(result_cpu):
        if len(dets) > 0:
            print(f"  Class {i}: {len(dets)} detections, max score: {dets[:, 4].max():.4f}")

# Now try GPU if available
if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("Testing on GPU...")
    print("=" * 60)
    try:
        model_gpu = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
        print("Model loaded on GPU successfully!")
        
        # Disable TTA
        model_gpu.cfg.data.test.pipeline[1]['img_scale'] = (800, 800)
        
        # Run inference on GPU
        print("Running inference on GPU...")
        result_gpu = inference_detector(model_gpu, IMAGE_FILE)
        
        print(f"\nGPU Result type: {type(result_gpu)}")
        if isinstance(result_gpu, list):
            total_gpu = sum(len(dets) for dets in result_gpu)
            print(f"Total detections on GPU: {total_gpu}")
            for i, dets in enumerate(result_gpu):
                if len(dets) > 0:
                    print(f"  Class {i}: {len(dets)} detections, max score: {dets[:, 4].max():.4f}")
    except Exception as e:
        print(f"GPU inference failed: {e}")

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)
