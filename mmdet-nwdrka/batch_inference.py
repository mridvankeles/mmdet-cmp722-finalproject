"""
Batch inference script for processing multiple images from a folder.
Saves visualization results to an output folder.
"""
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os
import glob
from pathlib import Path
from tqdm import tqdm


def main():
    # ==============================================
    # CONFIGURATION - Update these paths!
    # ==============================================
    CONFIG_FILE = 'workdir/faster_training_transformer/aitodv2train_faster_transformer.py'
    CHECKPOINT_FILE = 'workdir/faster_training_transformer/epoch_30.pth'
    
    # Input folder containing images
    INPUT_FOLDER = 'data/aitod_dataset/aitodv2/trainval'
    
    # Output folder for predictions
    OUTPUT_FOLDER = 'workdir/faster_training_transformer/batch_predictions'
    
    # Inference settings
    DEVICE = 'cuda:0'
    SCORE_THRESHOLD = 0.05
    PALETTE = 'coco'
    
    # Image extensions to process
    IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    
    # ==============================================
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Validate paths
    print("=" * 60)
    print("Batch Inference Configuration")
    print("=" * 60)
    print(f"Config: {CONFIG_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print(f"Input Folder: {INPUT_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print(f"Device: {DEVICE}")
    print(f"Score Threshold: {SCORE_THRESHOLD}")
    print("=" * 60)
    
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"❌ Config file not found: {CONFIG_FILE}")
    
    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"❌ Checkpoint file not found: {CHECKPOINT_FILE}")
    
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"❌ Input folder not found: {INPUT_FOLDER}")
    
    # Get all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"⚠️ No images found in {INPUT_FOLDER}")
        return
    
    print(f"\n✅ Found {len(image_files)} images to process")
    
    # Initialize the model
    print("\nLoading model...")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    
    # Override test pipeline to use simple single-scale preprocessing
    print("Configuring single-scale inference (replacing test pipeline)...")
    from mmcv import Config
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(800, 800),  # Single scale as tuple
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', 
                     mean=[123.675, 116.28, 103.53], 
                     std=[58.395, 57.12, 57.375], 
                     to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    model.cfg.data.test.pipeline = test_pipeline
    
    print("✅ Model loaded successfully!")
    
    # Process each image
    print(f"\n{'=' * 60}")
    print("Processing images...")
    print("=" * 60)
    
    total_detections = 0
    successful_images = 0
    
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing"), 1):
        try:
            # Get image filename
            image_name = os.path.basename(image_path)
            image_name_no_ext = os.path.splitext(image_name)[0]
            
            # Output path
            output_path = os.path.join(OUTPUT_FOLDER, f"{image_name_no_ext}_pred.jpg")
            
            # Run inference
            result = inference_detector(model, image_path)
            
            # Count detections
            if isinstance(result, list):
                num_dets = sum(len(dets) for dets in result)
                num_above_thresh = sum((dets[:, 4] >= SCORE_THRESHOLD).sum() for dets in result if len(dets) > 0)
            else:
                if hasattr(result, 'pred_instances'):
                    scores = result.pred_instances.scores.cpu().numpy()
                    num_dets = len(scores)
                    num_above_thresh = (scores >= SCORE_THRESHOLD).sum()
                else:
                    num_dets = 0
                    num_above_thresh = 0
            
            # Save visualization
            show_result_pyplot(
                model,
                image_path,
                result,
                palette=PALETTE,
                score_thr=SCORE_THRESHOLD,
                out_file=output_path
            )
            
            if num_above_thresh > 0:
                total_detections += num_above_thresh
                successful_images += 1
                tqdm.write(f"  [{idx}/{len(image_files)}] {image_name}: {num_above_thresh} detections → {output_path}")
            
        except Exception as e:
            tqdm.write(f"  ❌ Error processing {image_name}: {str(e)}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("Batch Inference Complete!")
    print("=" * 60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Images with detections: {successful_images}")
    print(f"Total detections (score >= {SCORE_THRESHOLD}): {total_detections}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("=" * 60)


if __name__ == '__main__':
    main()
