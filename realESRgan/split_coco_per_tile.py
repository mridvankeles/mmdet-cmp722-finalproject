import os
import json
from PIL import Image
import numpy as np
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

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

def batch_split_coco_per_tile(per_image_jsons_dir, images_dir, per_tile_jsons_dir, tile_size, stride, image_suffix):
    os.makedirs(per_tile_jsons_dir, exist_ok=True)
    json_files = [f for f in os.listdir(per_image_jsons_dir) if f.endswith(".json")]
    
    # Create one combined COCO JSON for all tiles
    combined_coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    annotation_id = 1
    image_id = 1
    
    for jf in json_files:
        with open(os.path.join(per_image_jsons_dir, jf), 'r') as f:
            jdata = json.load(f)
        image_info = jdata['images'][0]
        anns = jdata['annotations']
        categories = jdata['categories']
        
        # Set categories once
        if not combined_coco["categories"]:
            combined_coco["categories"] = categories
        
        # Process tiles for this image
        img_path = os.path.join(images_dir, image_info["file_name"].replace(".jpg", image_suffix))
        img = np.array(Image.open(img_path))
        height, width = img.shape[:2]
        tile_id = 1
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                tile_annotations_list = []
                
                for ann in anns:
                    bbox = ann["bbox"]  # [x_min, y_min, w, h]
                    bx, by, bw, bh = bbox
                    bx_end = bx + bw
                    by_end = by + bh
                    
                    # Check if bbox intersects with tile
                    if not (bx_end < x or bx > x_end or by_end < y or by > y_end):
                        # Clip bbox to tile boundaries
                        new_bx = max(0, bx - x)
                        new_by = max(0, by - y)
                        new_bx_end = min(tile_size, bx_end - x)
                        new_by_end = min(tile_size, by_end - y)
                        new_bw = new_bx_end - new_bx
                        new_bh = new_by_end - new_by
                        
                        if new_bw > 0 and new_bh > 0:
                            new_ann = ann.copy()
                            new_ann["bbox"] = [float(new_bx), float(new_by), float(new_bw), float(new_bh)]
                            new_ann["image_id"] = image_id
                            new_ann["id"] = annotation_id
                            
                            # Handle segmentation coordinates
                            if "segmentation" in ann and ann["segmentation"]:
                                new_segmentation = []
                                for seg in ann["segmentation"]:
                                    if isinstance(seg, list) and len(seg) >= 6:
                                        # Convert to list of (x, y) pairs
                                        coords = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                                        # Clip coordinates to tile boundaries
                                        clipped_coords = []
                                        for coord_x, coord_y in coords:
                                            # Transform to tile-relative coordinates
                                            tile_x = coord_x - x
                                            tile_y = coord_y - y
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
                            
                            tile_annotations_list.append(new_ann)
                            annotation_id += 1
                
                # Only add tile if it has annotations
                if tile_annotations_list:
                    tile_filename = f"{os.path.splitext(image_info['file_name'])[0]}_tile_{tile_id:05d}.{image_suffix}"
                    
                    # Add tile image info
                    tile_img_info = {
                        "id": image_id,
                        "file_name": tile_filename,
                        "height": tile_size,
                        "width": tile_size
                    }
                    combined_coco["images"].append(tile_img_info)
                    
                    # Add tile annotations
                    combined_coco["annotations"].extend(tile_annotations_list)
                    
                    image_id += 1
                
                tile_id += 1
    
    # Save the combined COCO JSON
    output_path = os.path.join(per_tile_jsons_dir, "all_tiles_coco.json")
    with open(output_path, "w") as f:
        json.dump(combined_coco, f, indent=2)
    
    print(f"Combined COCO JSON saved to: {output_path}")
    print(f"Total tiles with annotations: {len(combined_coco['images'])}")
    print(f"Total annotations: {len(combined_coco['annotations'])}")
    
    return combined_coco
