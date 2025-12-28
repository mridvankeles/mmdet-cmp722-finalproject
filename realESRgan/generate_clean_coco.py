import os
import json
import yaml
from pathlib import Path

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def is_empty_image_by_size(image_path, min_size_kb=1):
    """
    Check if an image is empty based on file size
    Empty/black images typically have very small file sizes
    
    Args:
        image_path: Path to the image file
        min_size_kb: Minimum file size in KB to consider non-empty (default: 1 KB)
    
    Returns:
        True if image is considered empty, False otherwise
    """
    try:
        # Get file size in bytes
        file_size_bytes = os.path.getsize(image_path)
        file_size_kb = file_size_bytes / 1024
        
        # Very small files are likely empty/black images
        if file_size_kb < min_size_kb:
            return True
            
        return False
        
    except Exception as e:
        print(f"Error checking file size for {image_path}: {e}")
        return True  # Consider problematic files as empty

def generate_clean_coco_json(config, min_file_size_kb=1):
    """
    Generate a clean COCO JSON file without empty images (detected by file size)
    
    Args:
        config: Configuration dictionary
        min_file_size_kb: Minimum file size in KB to consider non-empty
    """
    images_dir = config['tiles_dir']
    per_image_jsons_dir = config['per_image_jsons_dir']
    output_path = config['coco_json_path']
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Building footprint dataset without empty/black images (filtered by file size)",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "Clean COCO Generator",
            "date_created": "2024-01-01T00:00:00"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "building",
                "supercategory": "structure"
            }
        ]
    }
    
    # Get all JSON files
    json_files = [f for f in os.listdir(per_image_jsons_dir) if f.endswith('.json')]
    
    image_id = 1
    annotation_id = 1
    skipped_images = 0
    processed_images = 0
    total_size_processed = 0
    
    print(f"Processing {len(json_files)} JSON files...")
    print(f"Minimum file size threshold: {min_file_size_kb} KB")
    
    for json_file in json_files:
        json_path = os.path.join(per_image_jsons_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                image_data = json.load(f)
            
            # Get image info
            if 'images' not in image_data or not image_data['images']:
                continue
                
            image_info = image_data['images'][0]
            image_filename = image_info['file_name']
            
            # Check if corresponding image file exists
            image_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                skipped_images += 1
                continue
            
            # Check file size
            file_size_bytes = os.path.getsize(image_path)
            file_size_kb = file_size_bytes / 1024
            
            if is_empty_image_by_size(image_path, min_file_size_kb):
                print(f"Skipping small/empty image: {image_filename} ({file_size_kb:.2f} KB)")
                skipped_images += 1
                continue
            
            # Add image to COCO data
            new_image_info = {
                "id": image_id,
                "file_name": image_filename,
                "height": image_info.get('height', 800),
                "width": image_info.get('width', 800),
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": "2024-01-01T00:00:00"
            }
            coco_data["images"].append(new_image_info)
            
            # Add annotations
            if 'annotations' in image_data:
                for ann in image_data['annotations']:
                    new_annotation = ann.copy()
                    new_annotation["id"] = annotation_id
                    new_annotation["image_id"] = image_id
                    coco_data["annotations"].append(new_annotation)
                    annotation_id += 1
            
            processed_images += 1
            total_size_processed += file_size_kb
            image_id += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            skipped_images += 1
            continue
    
    # Save the clean COCO JSON
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nClean COCO JSON generated successfully!")
    print(f"Output file: {output_path}")
    print(f"Processed images: {processed_images}")
    print(f"Skipped images: {skipped_images}")
    print(f"Total images in COCO: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total size of processed images: {total_size_processed:.2f} KB")
    
    return coco_data

def analyze_file_sizes(images_dir, per_image_jsons_dir):
    """
    Analyze file sizes to help determine appropriate threshold
    """
    print("Analyzing file sizes to help determine threshold...")
    
    json_files = [f for f in os.listdir(per_image_jsons_dir) if f.endswith('.json')]
    file_sizes = []
    
    for json_file in json_files[:100]:  # Sample first 100 files
        try:
            with open(os.path.join(per_image_jsons_dir, json_file), 'r') as f:
                image_data = json.load(f)
            
            if 'images' in image_data and image_data['images']:
                image_filename = image_data['images'][0]['file_name']
                image_path = os.path.join(images_dir, image_filename)
                
                if os.path.exists(image_path):
                    file_size_kb = os.path.getsize(image_path) / 1024
                    file_sizes.append(file_size_kb)
        except:
            continue
    
    if file_sizes:
        file_sizes.sort()
        print(f"File size statistics (KB):")
        print(f"  Min: {min(file_sizes):.2f}")
        print(f"  Max: {max(file_sizes):.2f}")
        print(f"  Mean: {sum(file_sizes)/len(file_sizes):.2f}")
        print(f"  Median: {file_sizes[len(file_sizes)//2]:.2f}")
        print(f"  First quartile: {file_sizes[len(file_sizes)//4]:.2f}")
        print(f"  Third quartile: {file_sizes[3*len(file_sizes)//4]:.2f}")
        
        # Suggest threshold
        suggested_threshold = max(1, file_sizes[len(file_sizes)//4] * 0.5)
        print(f"  Suggested threshold: {suggested_threshold:.2f} KB")

def main():
    """Main function"""
    try:
        # Load configuration
        config = load_config()
        print("Configuration loaded successfully")
        
        # Analyze file sizes first
        analyze_file_sizes(config['tiles_dir'], config['per_tile_jsons_dir'])
        
        # Ask user for threshold
        print("\n" + "="*50)
        threshold_input = input("Enter minimum file size threshold in KB (default: 1): ").strip()
        
        if threshold_input:
            try:
                min_file_size_kb = float(threshold_input)
            except ValueError:
                print("Invalid input, using default threshold of 1 KB")
                min_file_size_kb = 1
        else:
            min_file_size_kb = 1
        
        print(f"Using threshold: {min_file_size_kb} KB")
        print("="*50)
        
        # Generate clean COCO JSON
        coco_data = generate_clean_coco_json(config, min_file_size_kb)
        
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
