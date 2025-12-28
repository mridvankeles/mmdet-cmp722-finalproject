import os
import json
import cv2
import numpy as np
from typing import Dict, List, Any

COLOR_GT = (255, 0, 0)      # Blue
COLOR_PRED = (0, 0, 255)    # Red

def load_coco_data(json_path: str, is_prediction: bool = False) -> Dict[str, Any]:
    """
    Loads COCO JSON file and structures data for easy lookup.
    Handles cases where the JSON data might be double-encoded as a string.
    """
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # --- FIX ---
    # If the loaded data is a string, it's likely double-encoded JSON.
    # Parse it one more time to get the actual list/dict.
    if isinstance(data, str):
        print("Data loaded as a string. Attempting to parse it as JSON...")
        data = json.loads(data)
    # --- END FIX ---

    if is_prediction:
        # Prediction files are typically a flat list of annotation-like objects
        annotations_list = data
        images_list = []
        categories_list = []
    else:
        # Ground truth files are a dictionary containing these keys
        annotations_list = data.get('annotations', [])
        images_list = data.get('images', [])
        categories_list = data.get('categories', [])

    image_id_to_filename = {img['id']: img['file_name'] for img in images_list}
    category_id_to_name = {cat['id']: cat['name'] for cat in categories_list}

    annotations_by_image_id = {}
    for ann in annotations_list:
        image_id = int(ann['image_id'])
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)
        
    return {
        'images': images_list,
        'categories': category_id_to_name,
        'annotations': annotations_by_image_id,
        'image_id_to_filename': image_id_to_filename
    }

def draw_images_with_gt_and_preds(
    image_folder: str,
    output_folder: str,
    images: List[Dict[str, Any]],
    gt_annotations: Dict[int, List[Dict[str, Any]]],
    pred_annotations: Dict[int, List[Dict[str, Any]]],
    id_to_label_map: Dict[int, str]
):
    """
    Draws GT and prediction boxes (with scores) for all images listed in `images`.
    """
    os.makedirs(output_folder, exist_ok=True)
    total_images = len(images)
    print(f"Found {total_images} images to process...\n")

    count = 0
    for img_info in images:
        image_id = img_info["id"]
        filename = img_info["file_name"]
        input_path = os.path.join(image_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if not os.path.exists(input_path):
            print(f"Skipping missing image: {input_path}")
            continue

        img = cv2.imread(input_path)
        if img is None:
            print(f"Could not read image: {input_path}")
            continue

        img_out = img.copy()

        # Draw GT boxes
        for gt in gt_annotations.get(image_id, []):
            x, y, w, h = map(int, gt["bbox"])
            label = id_to_label_map.get(gt["category_id"], "GT")
            cv2.rectangle(img_out, (x, y), (x + w, y + h), COLOR_GT, 2)
            cv2.putText(img_out, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2)

        # Draw Prediction boxes
        for pred in pred_annotations.get(image_id, []):
            x, y, w, h = map(int, pred["bbox"])
            label = id_to_label_map.get(pred["category_id"], "Pred")
            score = pred.get("score", None)
            text = f"{label} {score:.2f}" if score is not None else label
            cv2.rectangle(img_out, (x, y), (x + w, y + h), COLOR_PRED, 2)
            cv2.putText(img_out, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

        cv2.imwrite(output_path, img_out)
        count += 1
        print(f"[{count}/{total_images}] Saved: {output_path}")

    print(f"\n✅ Done. {count} images saved to: {output_folder}")

gt_json_path= "workdir/tuesday_nato/instances_default.json"
pred_json_path= "workdir/tuesday_nato/instances_default.json"
image_folder =  "data/tuesday_nato/images"
output_folder =  "data/tuesday_nato/images_out"

gt_data = load_coco_data(gt_json_path, is_prediction=False)
pred_data = load_coco_data(pred_json_path, is_prediction=False)

draw_images_with_gt_and_preds(
    image_folder=image_folder,
    output_folder=output_folder,
    images=gt_data["images"],
    gt_annotations=gt_data["annotations"],
    pred_annotations=pred_data["annotations"],
    id_to_label_map=gt_data["categories"]
)
