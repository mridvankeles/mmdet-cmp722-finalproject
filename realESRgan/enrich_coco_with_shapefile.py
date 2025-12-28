import os
import json
import yaml
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import pandas as pd

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_shapefile_data(shapefile_path):
    """
    Load shapefile data containing building height information
    """
    print(f"Loading shapefile from: {shapefile_path}")
    
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"Shapefile loaded successfully!")
        print(f"Total records: {len(gdf)}")
        print(f"Coordinate system: {gdf.crs}")
        print(f"Bounding box: {gdf.total_bounds}")
        
        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None

def load_coco_annotations(coco_path):
    """
    Load existing COCO annotations
    """
    print(f"Loading COCO annotations from: {coco_path}")
    
    try:
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"COCO data loaded successfully!")
        print(f"Total images: {len(coco_data.get('images', []))}")
        print(f"Total annotations: {len(coco_data.get('annotations', []))}")
        
        return coco_data
    except Exception as e:
        print(f"Error loading COCO data: {e}")
        return None

def load_original_coco(original_coco_path):
    """
    Load original COCO file to get image dimensions
    """
    print(f"Loading original COCO from: {original_coco_path}")
    
    try:
        with open(original_coco_path, 'r') as f:
            original_data = json.load(f)
        
        if original_data.get('images'):
            img_info = original_data['images'][0]
            print(f"Original image: {img_info['file_name']}")
            print(f"Dimensions: {img_info['width']}x{img_info['height']}")
            return original_data
        
        return None
    except Exception as e:
        print(f"Error loading original COCO: {e}")
        return None

def parse_tile_info(tile_filename):
    """
    Parse tile filename to get tile position
    Expected format: tile_00001.png, tile_00261.png, etc.
    Returns tile index (1-based)
    """
    import re
    match = re.search(r'tile_(\d+)\.png', tile_filename)
    if match:
        return int(match.group(1))
    return 1

def calculate_tile_coordinates(tile_index, original_width, original_height, tile_size, stride):
    """
    Calculate the coordinates of a tile in the original image
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

def transform_coordinates_to_shapefile(bbox, tile_x, tile_y, original_width, original_height, shapefile_bounds):
    """
    Transform tile-relative bbox coordinates to shapefile coordinate system
    """
    # Convert bbox from tile-relative to original image coordinates
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    
    # Transform to original image coordinates
    original_x = tile_x + bbox_x
    original_y = tile_y + bbox_y
    
    # Normalize to 0-1 range
    norm_x = original_x / original_width
    norm_y = original_y / original_height
    
    # Transform to shapefile coordinate system
    shapefile_x = shapefile_bounds[0] + norm_x * (shapefile_bounds[2] - shapefile_bounds[0])
    shapefile_y = shapefile_bounds[1] + norm_y * (shapefile_bounds[3] - shapefile_bounds[1])
    
    return shapefile_x, shapefile_y

def find_matching_buildings(annotation, tile_x, tile_y, original_width, original_height, 
                           shapefile_gdf, shapefile_bounds, tolerance=100):
    """
    Find matching building height data for a given annotation
    """
    bbox = annotation.get('bbox', [0, 0, 0, 0])
    
    # Transform coordinates to shapefile system
    center_x, center_y = transform_coordinates_to_shapefile(
        bbox, tile_x, tile_y, original_width, original_height, shapefile_bounds
    )
    
    # Create a search area around the transformed point
    search_box = box(center_x - tolerance, center_y - tolerance, 
                     center_x + tolerance, center_y + tolerance)
    
    # Find shapefile points within the search area
    matching_buildings = []
    
    for idx, row in shapefile_gdf.iterrows():
        point = row.geometry
        
        if search_box.contains(point):
            distance = point.distance(Point(center_x, center_y))
            matching_buildings.append({
                'height': row['NAME'],
                'elevation': row['ELEVATION'],
                'layer': row['LAYER'],
                'distance': distance
            })
    
    # Return the closest match if multiple found
    if matching_buildings:
        return min(matching_buildings, key=lambda x: x['distance'])
    
    return None

def enrich_coco_annotations(coco_data, original_coco_data, shapefile_gdf):
    """
    Enrich COCO annotations with building height information from shapefile
    """
    print("Enriching COCO annotations with building height data...")
    
    # Get original image dimensions
    original_width = original_coco_data['images'][0]['width']
    original_height = original_coco_data['images'][0]['height']
    tile_size = 300  # from config
    stride = 300      # from config
    
    # Get shapefile bounds
    shapefile_bounds = shapefile_gdf.total_bounds
    
    print(f"Original image: {original_width}x{original_height}")
    print(f"Tile size: {tile_size}, Stride: {stride}")
    print(f"Shapefile bounds: {shapefile_bounds}")
    
    enriched_annotations = []
    enriched_count = 0
    skipped_missing_attr = 0
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Process each image and its annotations
    for image_info in coco_data.get('images', []):
        image_id = image_info['id']
        tile_filename = image_info['file_name'].split('/')[-1]  # Get just filename
        
        # Parse tile position
        tile_index = parse_tile_info(tile_filename)
        tile_x, tile_y = calculate_tile_coordinates(tile_index, original_width, original_height, tile_size, stride)
        
        print(f"Processing image {image_id}: {tile_filename} at position ({tile_x}, {tile_y})")
        
        # Process annotations for this image
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                # Find matching building height data
                building_info = find_matching_buildings(
                    ann, tile_x, tile_y, original_width, original_height, 
                    shapefile_gdf, shapefile_bounds
                )
                
                # Create enriched annotation
                enriched_ann = ann.copy()
                
                # Keep only annotations that have valid height/elevation attributes
                if building_info and pd.notna(building_info['height']) and pd.notna(building_info['elevation']):
                    enriched_ann['building_height'] = {
                        'height_value': float(building_info['height']) if pd.notna(building_info['height']) else None,
                        'elevation': float(building_info['elevation']) if pd.notna(building_info['elevation']) else None,
                        'layer_type': building_info.get('layer'),
                        'source': 'shapefile',
                        'tile_position': [tile_x, tile_y],
                        'original_coords': [tile_x + ann['bbox'][0], tile_y + ann['bbox'][1]]
                    }
                    enriched_annotations.append(enriched_ann)
                    enriched_count += 1
                else:
                    skipped_missing_attr += 1
    
    print(f"Enriched {enriched_count} annotations with building height data")
    
    # Remove images without annotations and reindex image and annotation ids to keep consistency
    image_id_to_has_ann = {img['id']: False for img in coco_data.get('images', [])}
    for ann in enriched_annotations:
        image_id_to_has_ann[ann['image_id']] = True

    # Filter images
    kept_images = [img for img in coco_data.get('images', []) if image_id_to_has_ann.get(img['id'], False)]

    # Reindex images to have consecutive ids and update annotations accordingly
    old_to_new_image_id = {}
    for new_id, img in enumerate(kept_images, start=1):
        old_to_new_image_id[img['id']] = new_id
        img['id'] = new_id

    for ann in enriched_annotations:
        ann['image_id'] = old_to_new_image_id[ann['image_id']]

    # Reindex annotation ids
    for new_ann_id, ann in enumerate(enriched_annotations, start=1):
        ann['id'] = new_ann_id

    # Update COCO data
    enriched_coco = coco_data.copy()
    enriched_coco['images'] = kept_images
    enriched_coco['annotations'] = enriched_annotations
    
    # Add metadata about the enrichment
    enriched_coco['info']['enrichment'] = {
        'description': 'COCO annotations enriched with building height data from shapefile',
        'shapefile_source': 'istanbul_istek_alan_kot.shp',
        'enriched_annotations': enriched_count,
        'total_annotations': len(enriched_annotations),
        'skipped_due_to_missing_height_or_elevation': skipped_missing_attr,
        'coordinate_transformation': {
            'original_image': f"{original_width}x{original_height}",
            'tile_size': tile_size,
            'stride': stride,
            'shapefile_bounds': shapefile_bounds.tolist()
        }
    }
    
    return enriched_coco

def save_enriched_coco(enriched_coco, output_path):
    """
    Save enriched COCO data to file
    """
    print(f"Saving enriched COCO data to: {output_path}")
    
    try:
        with open(output_path, 'w') as f:
            json.dump(enriched_coco, f, indent=2)
        
        print(f"Enriched COCO data saved successfully!")
        return True
    except Exception as e:
        print(f"Error saving enriched COCO data: {e}")
        return False

def main():
    """Main function"""
    try:
        # Load configuration
        config = load_config()
        print("Configuration loaded successfully")
        
        # Paths
        shapefile_path = "dataset/istanbul_istek_alan_kot/istanbul_istek_alan_kot.shp"
        coco_path = "dataset/annotation_full/tiles_annotations/tiles_coco.json"
        original_coco_path = "dataset/annotation_full/istanbul.json"
        output_path = "dataset/annotation_full/tiles_annotations/tiles_coco_enriched_fixed.json"
        
        # Check if files exist
        if not os.path.exists(shapefile_path):
            print(f"Error: Shapefile not found at {shapefile_path}")
            return 1
        
        if not os.path.exists(coco_path):
            print(f"Error: COCO file not found at {coco_path}")
            return 1
        
        if not os.path.exists(original_coco_path):
            print(f"Error: Original COCO file not found at {original_coco_path}")
            return 1
        
        # Load shapefile data
        shapefile_gdf = load_shapefile_data(shapefile_path)
        if shapefile_gdf is None:
            return 1
        
        # Load COCO annotations
        coco_data = load_coco_annotations(coco_path)
        if coco_data is None:
            return 1
        
        # Load original COCO data
        original_coco_data = load_original_coco(original_coco_path)
        if original_coco_data is None:
            return 1
        
        # Enrich COCO annotations
        enriched_coco = enrich_coco_annotations(coco_data, original_coco_data, shapefile_gdf)
        
        # Save enriched data
        if save_enriched_coco(enriched_coco, output_path):
            print("\nScript completed successfully!")
            print(f"Enriched COCO file saved to: {output_path}")
            
            # Check results
            with open(output_path, 'r') as f:
                result_data = json.load(f)
            
            # Count enriched annotations
            enriched_count = sum(1 for ann in result_data['annotations'] 
                               if ann['building_height']['source'] == 'shapefile')
            
            print(f"Successfully enriched {enriched_count} annotations with building height data")
            
        else:
            print("\nScript failed to save enriched data!")
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())