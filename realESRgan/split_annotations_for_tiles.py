import os
import json
import yaml
from pathlib import Path

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_consolidated_coco_for_tiles(config):
    """
    Create one consolidated COCO JSON file containing all tiled images and their annotations
    """
    tiles_dir = config['tiles_dir']
    per_image_jsons_dir = config['per_image_jsons_dir']
    per_tile_jsons_dir = config['per_tile_jsons_dir']
    tile_size = config['tile_size']
    stride = config['stride']
    
    # Create output directory
    os.makedirs(per_tile_jsons_dir, exist_ok=True)
    
    # Initialize consolidated COCO structure
    consolidated_coco = {
        "info": {
            "description": "Consolidated COCO dataset with all tiled images and annotations",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "Tile Annotation Consolidator",
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
        "categories": []
    }
    
    # Get all tile folders
    tile_folders = [d for d in os.listdir(tiles_dir) if os.path.isdir(os.path.join(tiles_dir, d))]
    
    print(f"Found {len(tile_folders)} tile folders")
    
    image_id = 1
    annotation_id = 1
    total_tiles = 0
    total_annotations = 0
    
    # Process each tile folder
    for tile_folder in tile_folders:
        tile_folder_path = os.path.join(tiles_dir, tile_folder)
        tile_images = [f for f in os.listdir(tile_folder_path) if f.endswith('.png')]
        
        print(f"\nProcessing tile folder: {tile_folder}")
        print(f"  Found {len(tile_images)} tile images")
        
        # Find corresponding JSON file
        json_file = None
        for f in os.listdir(per_image_jsons_dir):
            if f.startswith(tile_folder) and f.endswith('.json'):
                json_file = f
                break
        
        if not json_file:
            print(f"  No corresponding JSON file found for {tile_folder}")
            continue
        
        json_path = os.path.join(per_image_jsons_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                image_data = json.load(f)
            
            # Set categories once (should be the same for all)
            if not consolidated_coco["categories"]:
                consolidated_coco["categories"] = image_data.get('categories', [])
            
            # Get original image dimensions
            if 'images' not in image_data or not image_data['images']:
                continue
                
            original_image = image_data['images'][0]
            original_width = original_image.get('width', 800)
            original_height = original_image.get('height', 800)
            
            # Calculate tile positions
            tiles_info = []
            for y in range(0, original_height, stride):
                for x in range(0, original_width, stride):
                    y_end = min(y + tile_size, original_height)
                    x_end = min(x + tile_size, original_width)
                    
                    # Find corresponding tile image
                    tile_filename = f"tile_{len(tiles_info)+1:05d}.png"
                    if tile_filename in tile_images:
                        tiles_info.append({
                            'x': x, 'y': y, 'x_end': x_end, 'y_end': y_end,
                            'filename': tile_filename,
                            'folder': tile_folder
                        })
            
            print(f"  Generated {len(tiles_info)} tile annotations")
            
            # Process each tile
            for i, tile_info in enumerate(tiles_info):
                tile_annotations = []
                
                # Process annotations for this tile
                if 'annotations' in image_data:
                    for ann in image_data['annotations']:
                        bbox = ann.get('bbox', [0, 0, 0, 0])
                        bx, by, bw, bh = bbox
                        bx_end = bx + bw
                        by_end = by + bh
                        
                        # Check if bbox intersects with tile
                        if not (bx_end < tile_info['x'] or bx > tile_info['x_end'] or 
                               by_end < tile_info['y'] or by > tile_info['y_end']):
                            
                            # Clip bbox to tile boundaries
                            new_bx = max(0, bx - tile_info['x'])
                            new_by = max(0, by - tile_info['y'])
                            new_bx_end = min(tile_size, bx_end - tile_info['x'])
                            new_by_end = min(tile_size, by_end - tile_info['y'])
                            new_bw = new_bx_end - new_bx
                            new_bh = new_by_end - new_by
                            
                            if new_bw > 0 and new_bh > 0:
                                new_ann = ann.copy()
                                new_ann["bbox"] = [float(new_bx), float(new_by), float(new_bw), float(new_bh)]
                                new_ann["image_id"] = image_id
                                new_ann["id"] = annotation_id
                                
                                # Handle segmentation coordinates if present
                                if "segmentation" in ann and ann["segmentation"]:
                                    new_segmentation = []
                                    for seg in ann["segmentation"]:
                                        if isinstance(seg, list) and len(seg) >= 6:
                                            # Convert to list of (x, y) pairs
                                            coords = [(seg[j], seg[j+1]) for j in range(0, len(seg), 2)]
                                            # Clip coordinates to tile boundaries
                                            clipped_coords = []
                                            for coord_x, coord_y in coords:
                                                # Transform to tile-relative coordinates
                                                tile_x = coord_x - tile_info['x']
                                                tile_y = coord_y - tile_info['y']
                                                # Clamp to tile boundaries
                                                tile_x = max(0, min(tile_size, tile_x))
                                                tile_y = max(0, min(tile_size, tile_y))
                                                clipped_coords.extend([tile_x, tile_y])
                                            
                                            # Only keep segmentation if it has enough valid points
                                            if len(clipped_coords) >= 6:
                                                new_segmentation.append(clipped_coords)
                                    
                                    if new_segmentation:
                                        new_ann["segmentation"] = new_segmentation
                                        # Recalculate area based on new segmentation
                                        new_ann["area"] = calculate_polygon_area(new_segmentation[0]) if new_segmentation else 0.0
                                    else:
                                        # Skip annotation if no valid segmentation remains
                                        continue
                                
                                tile_annotations.append(new_ann)
                                annotation_id += 1
                
                # Only add tile if it has annotations
                if tile_annotations:
                    # Add tile image info to consolidated COCO
                    tile_img_info = {
                        "id": image_id,
                        "file_name": f"{tile_info['folder']}/{tile_info['filename']}",
                        "height": tile_size,
                        "width": tile_size,
                        "license": 1,
                        "flickr_url": "",
                        "coco_url": "",
                        "date_captured": "2024-01-01T00:00:00"
                    }
                    consolidated_coco["images"].append(tile_img_info)
                    
                    # Add tile annotations to consolidated COCO
                    consolidated_coco["annotations"].extend(tile_annotations)
                    
                    total_tiles += 1
                    total_annotations += len(tile_annotations)
                    
                    print(f"    Added: {tile_info['folder']}/{tile_info['filename']} ({len(tile_annotations)} annotations)")
                
                image_id += 1
        
        except Exception as e:
            print(f"  Error processing {json_file}: {e}")
            continue
    
    # Save consolidated COCO JSON
    output_path = os.path.join(per_tile_jsons_dir, "consolidated_tiles_coco.json")
    with open(output_path, 'w') as f:
        json.dump(consolidated_coco, f, indent=2)
    
    print(f"\nConsolidated COCO JSON created successfully!")
    print(f"Output file: {output_path}")
    print(f"Total tiles: {total_tiles}")
    print(f"Total annotations: {total_annotations}")
    print(f"Total images in COCO: {len(consolidated_coco['images'])}")
    
    return consolidated_coco

def calculate_polygon_area(segmentation):
    """Calculate area of a polygon using shoelace formula"""
    if len(segmentation) < 6:  # Need at least 3 points (6 coordinates)
        return 0.0
    
    coords = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
    n = len(coords)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    
    return abs(area) / 2.0

def main():
    """Main function"""
    try:
        # Load configuration
        config = load_config()
        print("Configuration loaded successfully")
        
        # Create consolidated COCO for tiles
        consolidated_coco = create_consolidated_coco_for_tiles(config)
        
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
