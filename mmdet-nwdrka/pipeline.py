import cv2
import json
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm
from torchvision.ops import nms
import matplotlib.pyplot as plt
import pickle
import numpy as np
import json
from typing import List, Dict, Any
import os
import glob
import pickle
from mmdet.apis import init_detector, inference_detector
from typing import List, Dict, Any, Tuple
import numpy as np



def tile_images_and_get_metadata(
    input_dir: str, 
    cvat_xml: str, 
    output_dir: str, 
    tile_w: int = 300, 
    tile_h: int = 300, 
    stride: int = None
) -> Tuple[Dict[int, Tuple[str, int, int]], List[Dict[str, Any]], Dict[str, List[Dict]], Dict[str, int]]:
    """
    Tiles images based on CVAT XML, saves them to disk, and returns all necessary metadata.
    
    Returns:
        (offset_map, tiled_images_metadata, original_annotations_by_file, categories)
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    stride_w = stride if stride else tile_w
    stride_h = stride if stride else tile_h

    tree = ET.parse(cvat_xml)
    root = tree.getroot()

    categories = {}
    category_id_counter = 0 # MMDetection compatible index
    tiled_images_metadata = []
    original_annotations_by_file = {}
    offset_map = {}
    img_id = 1

    for image_tag in tqdm(root.findall("image"), desc="Tiling Images"):
        filename = image_tag.get("name")
        width = int(image_tag.get("width"))
        height = int(image_tag.get("height"))
        filepath = glob.glob(os.path.join(input_dir, "**", filename), recursive=True)
        if not filepath:
            continue
        filepath = filepath[0]

        img = cv2.imread(filepath)

        # 1. Collect Original Annotations and Categories
        bboxes = []
        for box in image_tag.findall("box"):
            label = box.get("label")
            if label not in categories:
                categories[label] = category_id_counter
                category_id_counter += 1
            xtl, ytl, xbr, ybr = map(float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")])
            bboxes.append({"label": label, "category_id": categories[label], "bbox": [xtl, ytl, xbr, ybr]})
        
        original_annotations_by_file[filename] = bboxes

        # 2. Sliding Window and Tiling
        for y in range(0, height, stride_h):
            for x in range(0, width, stride_w):
                crop = img[y:y+tile_h, x:x+tile_w]
                ch, cw = crop.shape[:2]

                # Pad if necessary
                if ch < tile_h or cw < tile_w:
                    padded = cv2.copyMakeBorder(
                        crop, 0, tile_h-ch, 0, tile_w-cw,
                        cv2.BORDER_CONSTANT, value=(0,0,0)
                    )
                    crop = padded

                new_filename = f"{img_id}.jpg"
                cv2.imwrite(os.path.join(output_dir, "images", new_filename), crop)

                # Store metadata linking tile to original image and offset
                tiled_images_metadata.append({
                    "id": img_id,
                    "file_name": new_filename,
                    "width": tile_w,
                    "height": tile_h,
                    "original_file": filename, 
                    "offset_x": x,
                    "offset_y": y
                })
                offset_map[img_id] = (filename, x, y)
                img_id += 1

    print(f"✅ Tiling complete. {len(tiled_images_metadata)} images saved.")
    return offset_map, tiled_images_metadata, original_annotations_by_file, categories

def clip_and_format_annotations(
    tiled_images_metadata: List[Dict[str, Any]], 
    original_annotations: Dict[str, List[Dict]], 
    categories: Dict[str, int], 
    output_dir: str
) -> Dict[str, Any]:
    """
    Clips original bounding boxes to tiled patches and formats them into COCO JSON.
    """
    annotations = []
    ann_id = 1
    
    # Use tile dimensions from the metadata (assuming uniformity)
    if not tiled_images_metadata:
        return {"images": [], "annotations": [], "categories": []}
    
    tile_w = tiled_images_metadata[0]['width']
    tile_h = tiled_images_metadata[0]['height']

    for img_meta in tqdm(tiled_images_metadata, desc="Clipping Annotations"):
        img_id = img_meta['id']
        filename = img_meta['original_file']
        off_x = img_meta['offset_x']
        off_y = img_meta['offset_y']
        
        bboxes_original = original_annotations.get(filename, [])

        for bb in bboxes_original:
            xtl, ytl, xbr, ybr = bb["bbox"]

            # Check for overlap
            if xbr < off_x or xtl > off_x + tile_w or ybr < off_y or ytl > off_y + tile_h:
                continue

            # Clip and translate coordinates relative to the tile's top-left (off_x, off_y)
            new_xmin = max(0, xtl - off_x)
            new_ymin = max(0, ytl - off_y)
            new_xmax = min(tile_w, xbr - off_x)
            new_ymax = min(tile_h, ybr - off_y)

            w = new_xmax - new_xmin
            h = new_ymax - new_ymin

            if w > 0 and h > 0:
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": bb["category_id"],
                    "bbox": [new_xmin, new_ymin, w, h],
                    "area": w*h,
                    "iscrowd": 0
                })
                ann_id += 1

    # Final COCO JSON structure
    coco_output = {
        # Select only the required keys for the final COCO image list
        "images": [{k: img_meta[k] for k in ['id', 'file_name', 'width', 'height']} for img_meta in tiled_images_metadata],
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }

    output_path = os.path.join(output_dir, "annotations.json")
    with open(output_path, "w") as f:
        json.dump(coco_output, f, indent=2)

    print(f"✅ Tiled COCO annotations saved to {output_path}")

    return coco_output

def convert_cvat_xml_to_coco_json(xml_file: str, output_json_file: str) -> Dict[str, Any]:
    """
    Converts a CVAT XML annotation file (for untiled data) to COCO JSON format.

    Args:
        xml_file (str): Path to the input CVAT XML file.
        output_json_file (str): Path to save the output COCO JSON file.

    Returns:
        Dict[str, Any]: The generated COCO JSON object (dictionary).
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error processing XML file: {e}")
        return {}

    coco_output: Dict[str, Any] = {
        "info": {"description": "Converted from CVAT XML to COCO JSON"},
        "licenses": [], "images": [], "annotations": [], "categories": []
    }

    category_map: Dict[str, int] = {}
    category_id_counter = 1  # COCO standard typically starts IDs at 1
    annotation_id_counter = 1
    
    print("Processing images and annotations...")

    for image_elem in tqdm(root.findall('image'), desc="Converting Images"):
        image_id = int(image_elem.get('id'))
        
        image_info = {
            "id": image_id,
            "width": int(image_elem.get('width')),
            "height": int(image_elem.get('height')),
            "file_name": os.path.basename(image_elem.get('name')),
            "license": 0, "flickr_url": "", "coco_url": "", "date_captured": ""
        }
        coco_output['images'].append(image_info)

        for box_elem in image_elem.findall('box'):
            label = box_elem.get('label')

            if label not in category_map:
                category_map[label] = category_id_counter
                coco_output['categories'].append({"id": category_id_counter, "name": label, "supercategory": "none"})
                category_id_counter += 1
            
            category_id = category_map[label]

            xtl, ytl, xbr, ybr = map(float, [box_elem.get('xtl'), box_elem.get('ytl'), box_elem.get('xbr'), box_elem.get('ybr')])

            width = xbr - xtl
            height = ybr - ytl
            bbox = [xtl, ytl, width, height]
            area = width * height

            annotation_info = {
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco_output['annotations'].append(annotation_info)
            annotation_id_counter += 1

    output_dir = os.path.dirname(output_json_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_file, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"\n✅ COCO JSON file saved to: {output_json_file}")
    return coco_output

def stitch_predictions(preds, offset_map, iou_thresh=0.5):
    """
    Stitches tiled predictions (in COCO result list format) back onto the original image 
    and applies Class-Aware NMS to remove duplicate detections.
    """
    stitched = {}

    # 1. Map tile predictions to global coordinates and group by original image file
    for p in preds:
        patch_id = p["image_id"]
        if patch_id not in offset_map:
            continue
        orig_file, off_x, off_y = offset_map[patch_id]

        x, y, w, h = p["bbox"]
        
        # Global XYXY coordinates
        gx1, gy1 = x + off_x, y + off_y
        gx2, gy2 = gx1 + w, gy1 + h

        if orig_file not in stitched:
            stitched[orig_file] = {"boxes": [], "scores": [], "labels": []}

        stitched[orig_file]["boxes"].append([gx1, gy1, gx2, gy2])
        stitched[orig_file]["scores"].append(p["score"])
        stitched[orig_file]["labels"].append(p["category_id"])

    final_results = {}
    
    # 2. Process each original image: Convert to tensor and apply Class-Aware NMS
    for fname, data in stitched.items():
        if not data["boxes"]:
            continue

        boxes = torch.tensor(data["boxes"], dtype=torch.float32)
        scores = torch.tensor(data["scores"])
        labels = torch.tensor(data["labels"])
        
        # List to collect NMS-filtered detections
        image_detections = []
        
        # --- Apply Class-Aware NMS ---
        # Iterate over unique classes present in the image
        for label_id in torch.unique(labels):
            # 1. Filter tensors to only include current class
            class_mask = (labels == label_id)
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            if class_boxes.shape[0] == 0:
                continue
                
            # 2. Apply NMS (class-specific)
            # This 'keep' tensor holds the indices of the boxes to keep for the current class
            keep_indices = nms(class_boxes, class_scores, iou_thresh)
            
            # 3. Collect the filtered boxes, scores, and labels
            final_boxes_class = class_boxes[keep_indices].numpy().tolist()
            final_scores_class = class_scores[keep_indices].numpy().tolist()
            
            for b, s in zip(final_boxes_class, final_scores_class):
                x1, y1, x2, y2 = b
                
                # Convert back to [x, y, w, h] COCO format
                image_detections.append({
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "score": s,
                    "category_id": int(label_id.item()) # Ensure category_id is correct integer
                })

        final_results[fname] = image_detections

    return final_results

def load_and_verify_predictions(pickle_path: str) -> List[List[np.ndarray]]:
    """Loads the pickled MMDetection V2.x results."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Prediction file not found: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, list) or not all(isinstance(img_preds, list) for img_preds in data):
        print("Warning: Expected MMDetection V2.x list[list[np.ndarray]]. Check file format.")
    
    return data

def final_preds_to_cvat_xml(final_preds: Dict[str, List[Dict]], coco_ann_data: Dict[str, Any], category_label_map: Dict[int, str], output_xml_file: str):
    """
    Converts stitched predictions into a single CVAT/Pascal VOC XML file.

    Args:
        final_preds (Dict): Stitched results: {filename: List[{bbox:[x,y,w,h], score, category_id}]}.
        coco_ann_data (Dict): The dictionary of the original COCO annotations (for image metadata).
        category_label_map (Dict): Map from MMDetection category ID (int) to label name (str).
        output_xml_file (str): Path to save the output CVAT XML file.
    """
    
    # Create mappings for quick lookup
    image_map = {img['file_name']: img for img in coco_ann_data.get('images', [])}
    
    # --- Start XML Generation ---
    root = ET.Element("annotations")
    
    # Process each image to create a <image> tag
    for file_name, preds in final_preds.items():
        if file_name not in image_map:
            print(f"Warning: Image '{file_name}' not found in COCO annotations. Skipping XML entry.")
            continue
            
        img_meta = image_map[file_name]
        
        # CVAT <image> ID should ideally match the ID used during CVAT data export
        # We use the COCO image ID here.
        image_tag = ET.SubElement(root, "image")
        image_tag.set("id", str(img_meta.get('id', 0))) 
        image_tag.set("name", file_name)
        image_tag.set("width", str(img_meta.get("width", 0)))
        image_tag.set("height", str(img_meta.get("height", 0)))
        
        # Add predictions (boxes) for this image
        for pred in preds:
            x, y, w, h = pred['bbox']
            
            # Convert [x, y, w, h] to [xtl, ytl, xbr, ybr] integer coords for XML
            xtl, ytl = x, y
            xbr, ybr = x + w, y + h
            
            category_name = category_label_map.get(pred['category_id'], "unknown")
            score = pred['score']
            
            box_tag = ET.SubElement(image_tag, "box")
            box_tag.set("label", category_name)
            box_tag.set("xtl", str(int(xtl)))
            box_tag.set("ytl", str(int(ytl)))
            box_tag.set("xbr", str(int(xbr)))
            box_tag.set("ybr", str(int(ybr)))
            box_tag.set("occluded", "0") 
            
            # Add the confidence score as an attribute for CVAT autolabeling
            ET.SubElement(box_tag, "attribute", name="score").text = f"{score:.4f}"

    # Write the XML to file
    tree = ET.ElementTree(root)
    tree.write(output_xml_file, xml_declaration=True, encoding="utf-8")
    print(f"✅ Successfully generated CVAT/Pascal VOC XML with {len(root)} images to: {output_xml_file}")

def convert_to_coco_list(mmdet_results: List[List[np.ndarray]]) -> List[Dict[str, Any]]:
    """
    Converts MMDetection V2.x output (list[list[np.ndarray]]) to a COCO result list format.
    
    The MMDetection V2.x format is:
    [
        [class0_preds_np_array, class1_preds_np_array, ...], # for image 1
        [class0_preds_np_array, class1_preds_np_array, ...], # for image 2
        ...
    ]
    Each np_array is N x 5: [x1, y1, x2, y2, score].
    """
    
    coco_list = []
    
    # Iterate through each image's predictions
    for image_id, img_preds_per_class in enumerate(mmdet_results, start=1):
        
        # Iterate through predictions for each category (class)
        # We ensure category_id starts at 0, matching your requirement and MMDetection's internal index
        for category_id, class_detections in enumerate(img_preds_per_class, start=0):
            
            # category_id = 0 corresponds to the first array in img_preds_per_class
            
            if class_detections.size == 0:
                continue

            # Process all detections for the current class
            for detection in class_detections:
                x1, y1, x2, y2, score = detection
                
                # Convert [x1, y1, x2, y2] to COCO [x, y, w, h] format
                w = x2 - x1
                h = y2 - y1
                bbox_coco = [float(x1), float(y1), float(w), float(h)]

                # Append to the final list
                coco_list.append({
                    "image_id": image_id,
                    "bbox": bbox_coco,
                    "score": float(score),
                    "category_id": category_id # This will be 0, 1, 2...
                })
                
    print(f"Successfully converted {len(coco_list)} total detections.")
    return coco_list

def final_preds_to_coco_results(final_preds: Dict[str, List[Dict]], coco_ann_data: Dict[str, Any], output_json_file: str) -> List[Dict]:
    """
    Converts stitched prediction results (grouped by filename) into a flat COCO result list 
    and saves it to a JSON file.

    Args:
        final_preds (Dict): Stitched results: {filename: List[{bbox:[x,y,w,h], score, category_id}]}.
        coco_ann_data (Dict): The dictionary of the original COCO annotations (to get image IDs).
        output_json_file (str): Path to save the COCO results JSON.
    
    Returns:
        List[Dict]: The flat list of COCO detection results.
    """
    coco_results = []
    
    # Create mapping from filename to COCO image ID
    filename_to_id = {img['file_name']: img['id'] for img in coco_ann_data['images']}
    
    # We must use the category IDs defined in the COCO annotations
    category_name_to_id = {cat['name']: cat['id'] for cat in coco_ann_data['categories']}
    
    detection_id = 0
    for file_name, detections in final_preds.items():
        if file_name not in filename_to_id:
            print(f"Warning: Skipping {file_name}. Not found in COCO annotations.")
            continue
            
        image_id = filename_to_id[file_name]
        
        for d in detections:
            # Note: d['category_id'] is the MMDetection index (0, 1, 2...). 
            # We must convert this back to the COCO category ID (1, 2, 3...) using the categories map.
            
            # Since the 'categories' dictionary in your script maps: {label_name: mmdet_id},
            # we need to ensure the correct COCO ID is used.
            
            # --- Assuming categories dictionary {label_name: mmdet_id} is available globally ---
            # For simplicity, we assume your category mapping is consistent (mmdet_id=0 maps to class1, etc.)
            
            # A more robust way would be to pass the category ID mapping used during training.
            
            # For now, we'll rely on the integer category_id returned by MMDetection.
            # *If your COCO IDs start at 1, you might need d['category_id'] + 1.*
            # Since your tiler starts category_id at 0, we'll assume the MMDET predictions
            # and the COCO result file should use the MMDET indices (0, 1, 2...) for now.
            
            # Use the MMDetection index directly as category_id for now (common in MMDET result files)
            coco_results.append({
                "image_id": image_id,
                "category_id": d['category_id'], 
                "bbox": [float(val) for val in d['bbox']], # [x, y, w, h] float
                "score": float(d['score'])
            })
            detection_id += 1

    # Save to JSON file
    with open(output_json_file, 'w') as f:
        json.dump(coco_results, f)
        
    print(f"\n✅ Saved {len(coco_results)} predictions to COCO JSON result file: {output_json_file}")
    return coco_results

def draw_comparison_results(
    image_path: str, 
    predictions: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]], 
    id_to_label_map: Dict[int, str], 
    save_path: str = None
):
    """
    Draws Ground Truth and Prediction bounding boxes on the same image.
    Saves the result to a specified path without displaying. Includes legend.

    Args:
        image_path (str): Path to the original image file.
        predictions (List[Dict]): List of stitched predictions ({bbox, score, category_id}).
        ground_truth (List[Dict]): List of ground truth annotations.
        id_to_label_map (Dict[int, str]): Dictionary mapping category_id (int) to label name (str).
        save_path (str): Path to save the resulting image. If None, image is not saved.
    """
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not read image at {image_path}")
        return

    img_draw_preds_gt = img_orig.copy() # Image with both preds and GT
    img_only_orig = img_orig.copy() # For the left side if side-by-side desired

    # --- 1. Draw Ground Truth (GT) in BLUE on the composite image ---
    GT_COLOR = (255, 0, 0) # Blue (B, G, R)
    for gt in ground_truth:
        x, y, w, h = map(int, gt["bbox"])
        label = id_to_label_map.get(gt["category_id"], "Unknown GT")
        cv2.rectangle(img_draw_preds_gt, (x, y), (x + w, y + h), GT_COLOR, 3) # Thicker line for GT
        cv2.putText(img_draw_preds_gt, label, (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_COLOR, 2)

    # --- 2. Draw Predictions (Pred) in RED on the composite image ---
    PRED_COLOR = (0, 0, 255) # Red (B, G, R)
    for pred in predictions:
        x, y, w, h = map(int, pred["bbox"])
        score = pred["score"]
        label = id_to_label_map.get(pred["category_id"], "Unknown Pred")
        cv2.rectangle(img_draw_preds_gt, (x, y), (x + w, y + h), PRED_COLOR, 2)
        cv2.putText(img_draw_preds_gt, f"{label} {score:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, PRED_COLOR, 2)
    
    # --- 3. Create Side-by-Side Image with Legend ---
    # Concatenate original image with the drawn image
    side_by_side = cv2.hconcat([img_only_orig, img_draw_preds_gt])

    # Create a simple legend image
    legend_height = 100
    legend = np.zeros((legend_height, side_by_side.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "Legend:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(legend, "Ground Truth (BLUE)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_COLOR, 2)
    cv2.putText(legend, "Prediction (RED)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PRED_COLOR, 2)
    
    # Concatenate the combined image and the legend
    final_image_with_legend = cv2.vconcat([side_by_side, legend])

    # --- 4. Save Result ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, final_image_with_legend)
        print(f"Saved comparison image to: {save_path}") # Optional: for verbose output



if __name__ == "__main__":
    # Örnek kullanım (Example Usage)
    INPUT_IMAGE_FOLDER = 'data/mine_deneme/images'
    INPUT_ANNOTATION_PATH = "data/mine_deneme/annotations.xml"
    
    # New File Paths for Output Predictions
    PREDICTION_COCO_JSON = 'workdir/faster_autolabel_mine_overfit_full/predictions_stitched_coco.json'
    PREDICTION_CVAT_XML = 'workdir/faster_autolabel_mine_overfit_full/predictions_stitched_cvat.xml'
    
    # GT file paths
    cvat_coco = "data/mine_deneme/annotations.json" 
    output_dir = "data/mine_deneme/dataset_tiled_overlap"
    
    CONFIG_FILE = 'workdir/faster_training_deneme_mine_overfit/aitodv2train_faster_r50_nwdrka_1x.py' 
    CHECKPOINT_FILE = 'workdir/faster_training_deneme_mine_overfit/latest.pth'
    PREDICTION_PICKLE_FILE = 'workdir/faster_autolabel_mine_overfit_full/unlabeled_predictions.pkl'
    AUTOLABEL_OUTPUT_JSON = 'workdir/faster_autolabel_mine_overfit_full/new_autolabeled_coco.json'

    tile_width = 800
    tile_height= 800
    stride = 200

    # 1. TILE IMAGES AND GET METADATA
    
    offset_map, tiled_meta, original_anns, categories = tile_images_and_get_metadata(
    INPUT_IMAGE_FOLDER, INPUT_ANNOTATION_PATH, output_dir, tile_w=tile_width, tile_h=tile_height, stride=stride
    )

    coco_json_data = clip_and_format_annotations(
        tiled_meta, original_anns, categories, output_dir
    )

    # Convert cvat xml to coco json (ORGINAL SIZED)
    coco_gt = convert_cvat_xml_to_coco_json(INPUT_ANNOTATION_PATH, cvat_coco)

    
    raw_predictions = load_and_verify_predictions(PREDICTION_PICKLE_FILE)
    
        
    prediction_list = convert_to_coco_list(raw_predictions)

    # Stitch Predictions
    final_preds = stitch_predictions(prediction_list, offset_map, iou_thresh=0.5)


    
    id_to_label_map = {v: k for k,v in categories.items()}
    final_preds_to_coco_results(final_preds, coco_gt, PREDICTION_COCO_JSON)
    final_preds_to_cvat_xml(final_preds, coco_gt, id_to_label_map, PREDICTION_CVAT_XML)
    
    # ----------------------------------------------------
    
    """# 4) Yan yana görselleştir (Compare GT vs. Predictions)
    # The GT loading logic is still implicit in the previous version, we'll keep the
    # visualization block simple by using the coco_gt structure for metadata.
    
    # Re-map coco_gt to be keyed by filename for visualization simplicity
    gt_by_filename = {img['file_name']: [] for img in coco_gt['images']}
    for ann in coco_gt['annotations']:
        # Find filename using image_id
        img_id = ann['image_id']
        file_name = next(img['file_name'] for img in coco_gt['images'] if img['id'] == img_id)
        gt_by_filename[file_name].append(ann)
    
    # Define a directory for saving comparison images
    COMPARISON_OUTPUT_FOLDER = "workdir/faster_testing_deneme_mine_overfit/comparison_images"
    os.makedirs(COMPARISON_OUTPUT_FOLDER, exist_ok=True) # Ensure folder exists
    
    for fname, preds in final_preds.items():
        orig_path = glob.glob(os.path.join(input_dir, "**", fname), recursive=True)[0]
        
        gt_boxes = gt_by_filename.get(fname, [])
        
        # Construct the save path for each image
        comparison_save_path = os.path.join(COMPARISON_OUTPUT_FOLDER, f"comparison_{fname}")
        
        draw_comparison_results(
            image_path=orig_path, 
            predictions=preds,
            ground_truth=gt_boxes,
            id_to_label_map=id_to_label_map,
            save_path=comparison_save_path # Pass the save path here
        )"""