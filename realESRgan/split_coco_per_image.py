import os
import json
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

#her bir görsel için json oluşturma 
def split_coco_per_image(coco_json_path, per_image_jsons_dir):
    os.makedirs(per_image_jsons_dir, exist_ok=True)
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    image_dict = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)
    for image_id, image_info in image_dict.items():
        filename = image_info['file_name']
        out_json = {
            "images": [image_info],
            "annotations": annotations_by_image.get(image_id, []),
            "categories": coco_data['categories']
        }
        out_path = os.path.join(per_image_jsons_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(out_path, 'w') as f:
            json.dump(out_json, f)
