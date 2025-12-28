import os
import json
import yaml
import re

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def parse_tile_position(tile_filename):
    """
    Parse tile filename to get tile position
    Expected format: tile_00001.png, tile_00002.png, etc.
    Returns tile index (1-based)
    """
    match = re.search(r'tile_(\d+)\.png', tile_filename)
    if match:
        return int(match.group(1))
    return 1

def calculate_tile_coordinates(tile_index, original_width, original_height, tile_size, stride):
    """
    Calculate the coordinates of a tile based on its index
    """
    # Calculate how many tiles fit in each row and column
    tiles_per_row = (original_width + stride - 1) // stride
    tiles_per_col = (original_height + stride - 1) // stride
    
    # Calculate tile position
    row = (tile_index - 1) // tiles_per_row
    col = (tile_index - 1) % tiles_per_row
    
    x = col * stride
    y = row * stride
    
    return x, y

def split_coco_for_tiles(config):
    """
    Split COCO annotations to match tiled images using config paths
    """
    # Load the original COCO annotation file from your config
    coco_json_path = config['coco_json_path']  # "dataset/annotation_full/istanbul.json"
    tiles_dir = config['sr_tiles_dir']            # "dataset/tiles"
    annotation_full_dir = config['annotation_full']  # "dataset/annotation_full"
    
    print(f"Loading COCO annotations from: {coco_json_path}")
    
    # Check if the COCO file exists
    if not os.path.exists(coco_json_path):
        print(f"Error: COCO file not found at {coco_json_path}")
        return None
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory in annotation_full folder
    output_dir = os.path.join(annotation_full_dir, "tiles_annotations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all tile folders from tiles_dir
    tile_folders = [d for d in os.listdir(tiles_dir) if os.path.isdir(os.path.join(tiles_dir, d))]
    print(f"Found {len(tile_folders)} tile folders in {tiles_dir}")
    
    # Create one consolidated COCO JSON for all tiles
    tiles_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    image_id = 1
    annotation_id = 1
    tile_size = config['tile_size']  # 300
    stride = config['stride']         # 300
    
    # Get original image dimensions from the first image in COCO data
    original_width = 0
    original_height = 0
    if coco_data.get('images'):
        original_width = coco_data['images'][0].get('width', 800)
        original_height = coco_data['images'][0].get('height', 800)
    
    print(f"Original image dimensions: {original_width}x{original_height}")
    print(f"Tile size: {tile_size}, Stride: {stride}")
    
    # Process each tile folder
    for tile_folder in tile_folders:
        tile_folder_path = os.path.join(tiles_dir, tile_folder)
        tile_images = [f for f in os.listdir(tile_folder_path) if f.endswith('.png')]
        
        print(f"Processing {tile_folder}: {len(tile_images)} tiles")
        
        # Add each tile image to the COCO data
        for tile_filename in tile_images:
            # Parse tile position
            tile_index = parse_tile_position(tile_filename)
            tile_x, tile_y = calculate_tile_coordinates(tile_index, original_width, original_height, tile_size, stride)
            
            # Add image info
            tile_img_info = {
                "id": image_id,
                "file_name": f"{tile_folder}/{tile_filename}",
                "height": 800,
                "width": 800,
            }
            tiles_coco["images"].append(tile_img_info)
            
            # Find annotations that intersect with this specific tile
            if 'annotations' in coco_data:
                for ann in coco_data['annotations']:
                    bbox = ann.get('bbox', [0, 0, 0, 0])
                    bx, by, bw, bh = bbox
                    bx_end = bx + bw
                    by_end = by + bh
                    
                    # Check if bbox intersects with tile
                    if not (bx_end < tile_x or bx > tile_x + tile_size or 
                           by_end < tile_y or by > tile_y + tile_size):
                        
                        # Clip bbox to tile boundaries
                        new_bx = max(0, bx - tile_x)
                        new_by = max(0, by - tile_y)
                        new_bx_end = min(tile_size, bx_end - tile_x)
                        new_by_end = min(tile_size, by_end - tile_y)
                        new_bw = new_bx_end - new_bx
                        new_bh = new_by_end - new_by
                        
                        if new_bw > 0 and new_bh > 0:
                            new_ann = ann.copy()
                            new_ann["bbox"] = [int(new_bx), int(new_by), int(new_bw), int(new_bh)]
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
                                            tile_coord_x = coord_x - tile_x
                                            tile_coord_y = coord_y - tile_y
                                            # Clamp to tile boundaries
                                            tile_coord_x = max(0, min(tile_size, tile_coord_x))
                                            tile_coord_y = max(0, min(tile_size, tile_coord_y))
                                            clipped_coords.extend([int(tile_coord_x), int(tile_coord_y)])
                                        
                                        # Only keep segmentation if it has enough valid points
                                        if len(clipped_coords) >= 6:
                                            new_segmentation.append(clipped_coords)
                                
                                if new_segmentation:
                                    new_ann["segmentation"] = new_segmentation
                                    # Recalculate area based on new segmentation
                                    #new_ann["area"] = calculate_polygon_area(new_segmentation[0]) if new_segmentation else 0.0
                                else:
                                    # Skip annotation if no valid segmentation remains
                                    continue
                            
                            tiles_coco["annotations"].append(new_ann)
                            annotation_id += 1
            
            image_id += 1
            
    output_path = os.path.join(config['annotation_full'], "tiles_coco.json")

    with open(output_path, 'w') as f:
        json.dump(tiles_coco, f)
    print(f"\nTiles COCO JSON saved to: {output_path}")
    
    return tiles_coco

"""def calculate_polygon_area(segmentation):
    #Calculate area of a polygon using shoelace formula
    if len(segmentation) < 6:  # Need at least 3 points (6 coordinates)
        return 0.0
    
    coords = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
    n = len(coords)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    
    return abs(area) / 2.0"""

def main():
    
    config = load_config()
    print("Configuration loaded successfully")
    print(f"Using paths from config:")
    print(f"  COCO source: {config['coco_json_path']}")
    print(f"  Tiles directory: {config['sr_tiles_dir']}")
    print(f"  Annotation output: {config['annotation_full']}")
    
    print(config)
    tiles_coco = split_coco_for_tiles(config)
        
    
    print("tiles_json:",tiles_coco)
    # Save the tiles COCO JSON in your annotation_full directory
    
    
    print(f"Total tile images: {len(tiles_coco['images'])}")
    print(f"Total annotations: {len(tiles_coco['annotations'])}")

if __name__ == "__main__":
    exit(main())
