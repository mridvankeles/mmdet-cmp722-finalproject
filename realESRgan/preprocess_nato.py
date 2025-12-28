```python
import os
import cv2
import glob
import json
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm
from torchvision.ops import nms
import matplotlib.pyplot as plt

def cvat_to_coco_tiled(input_dir, cvat_xml, output_dir, tile_w=300, tile_h=300, stride=None):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    stride_w = stride if stride else tile_w
    stride_h = stride if stride else tile_h

    tree = ET.parse(cvat_xml)
    root = tree.getroot()

    categories = {}
    category_id = 1
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    offset_map = {}  # patch_id -> (file, offset_x, offset_y)

    for image_tag in tqdm(root.findall("image")):
        filename = image_tag.get("name")
        width = int(image_tag.get("width"))
        height = int(image_tag.get("height"))
        filepath = glob.glob(os.path.join(input_dir, "**", filename), recursive=True)
        if not filepath:
            continue
        filepath = filepath[0]

        img = cv2.imread(filepath)

        # annotationları oku
        bboxes = []
        for box in image_tag.findall("box"):
            label = box.get("label")
            if label not in categories:
                categories[label] = category_id
                category_id += 1
            xtl, ytl, xbr, ybr = map(float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")])
            bboxes.append({"label": label, "bbox": [xtl, ytl, xbr, ybr]})

        # sliding window
        for y in range(0, height, stride_h):
            for x in range(0, width, stride_w):
                crop = img[y:y+tile_h, x:x+tile_w]
                ch, cw = crop.shape[:2]

                # pad gerekiyorsa
                if ch < tile_h or cw < tile_w:
                    padded = cv2.copyMakeBorder(
                        crop, 0, tile_h-ch, 0, tile_w-cw,
                        cv2.BORDER_CONSTANT, value=(0,0,0)
                    )
                    crop = padded

                new_filename = f"{img_id}.jpg"
                cv2.imwrite(os.path.join(output_dir, "images", new_filename), crop)

                images.append({
                    "id": img_id,
                    "file_name": new_filename,
                    "width": tile_w,
                    "height": tile_h
                })
                offset_map[img_id] = (filename, x, y)

                # bboxları uyarlama
                for bb in bboxes:
                    xtl, ytl, xbr, ybr = bb["bbox"]

                    if xbr < x or xtl > x+tile_w or ybr < y or ytl > y+tile_h:
                        continue  # tamamen dışarıda

                    new_xmin = max(0, xtl - x)
                    new_ymin = max(0, ytl - y)
                    new_xmax = min(tile_w, xbr - x)
                    new_ymax = min(tile_h, ybr - y)

                    w = new_xmax - new_xmin
                    h = new_ymax - new_ymin

                    if w > 0 and h > 0:
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": categories[bb["label"]],
                            "bbox": [new_xmin, new_ymin, w, h],
                            "area": w*h,
                            "iscrowd": 0
                        })
                        ann_id += 1

                img_id += 1

    # COCO json yaz
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }

    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ Done. {len(images)} tiled images saved to {output_dir}/images/")
    print(f"✅ COCO annotations saved to {output_dir}/annotations.json")

    return offset_map, categories


def stitch_predictions(preds, offset_map, iou_thresh=0.5):
    """
    preds: [
        {"image_id": patch_id, "bbox": [x,y,w,h], "score": float, "category_id": int}
    ]
    offset_map: patch_id -> (orig_filename, offset_x, offset_y)
    """
    stitched = {}

    for p in preds:
        patch_id = p["image_id"]
        if patch_id not in offset_map:
            continue
        orig_file, off_x, off_y = offset_map[patch_id]

        x, y, w, h = p["bbox"]
        gx, gy = x + off_x, y + off_y

        if orig_file not in stitched:
            stitched[orig_file] = {"boxes": [], "scores": [], "labels": []}

        stitched[orig_file]["boxes"].append([gx, gy, gx+w, gy+h])
        stitched[orig_file]["scores"].append(p["score"])
        stitched[orig_file]["labels"].append(p["category_id"])

    final_results = {}
    for fname, data in stitched.items():
        boxes = torch.tensor(data["boxes"], dtype=torch.float32)
        scores = torch.tensor(data["scores"])
        labels = torch.tensor(data["labels"])

        keep = nms(boxes, scores, iou_thresh)

        final_boxes = boxes[keep].numpy().tolist()
        final_scores = scores[keep].numpy().tolist()
        final_labels = labels[keep].numpy().tolist()

        final_results[fname] = []
        for b, s, l in zip(final_boxes, final_scores, final_labels):
            x1, y1, x2, y2 = b
            final_results[fname].append({
                "bbox": [x1, y1, x2-x1, y2-y1],
                "score": s,
                "category_id": l
            })
    return final_results


def draw_results_side_by_side(image_path, results, categories, save_path=None):
    img_orig = cv2.imread(image_path)
    img_draw = img_orig.copy()

    for r in results:
        x, y, w, h = map(int, r["bbox"])
        label = categories[r["category_id"]]
        score = r["score"]
        cv2.rectangle(img_draw, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img_draw, f"{label} {score:.2f}", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    side_by_side = cv2.hconcat([img_orig, img_draw])

    if save_path:
        cv2.imwrite(save_path, side_by_side)

    # matplotlib ile göster
    plt.figure(figsize=(16,8))
    plt.imshow(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def convert_cvat_to_coco(xml_file, output_json_file):
    """
    Converts a CVAT XML annotation file to COCO JSON format.

    This script handles bounding box annotations (<box> tags). It automatically
    discovers categories from the labels in the XML file.

    Args:
        xml_file (str): Path to the input CVAT XML file.
        output_json_file (str): Path to save the output COCO JSON file.
    """
    # XML dosyasını ayrıştır (parse et)
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return
    except FileNotFoundError:
        print(f"Error: Input file not found at {xml_file}")
        return

    # COCO formatı için temel yapıyı oluştur
    coco_output = {
        "info": {
            "description": "Converted from CVAT XML to COCO JSON",
            "version": "1.0",
            "year": 2025,
            "contributor": "CVAT to COCO Converter",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}
    category_id_counter = 1
    annotation_id_counter = 1
    
    print("Processing images and annotations...")

    # XML'deki her bir <image> etiketi için döngü başlat
    # tqdm ile bir ilerleme çubuğu ekleniyor
    for image_elem in tqdm(root.findall('image'), desc="Converting Images"):
        image_id = int(image_elem.get('id'))
        
        # COCO için resim bilgilerini oluştur
        image_info = {
            "id": image_id,
            "width": int(image_elem.get('width')),
            "height": int(image_elem.get('height')),
            "file_name": os.path.basename(image_elem.get('name')),
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_output['images'].append(image_info)

        # Resim içindeki her bir <box> etiketi için döngü
        for box_elem in image_elem.findall('box'):
            label = box_elem.get('label')

            # Eğer kategori daha önce görülmediyse, yeni bir kategori oluştur
            if label not in category_map:
                category_map[label] = category_id_counter
                category_info = {
                    "id": category_id_counter,
                    "name": label,
                    "supercategory": "none"
                }
                coco_output['categories'].append(category_info)
                category_id_counter += 1
            
            category_id = category_map[label]

            # CVAT koordinatlarını al (xtl, ytl, xbr, ybr)
            xtl = float(box_elem.get('xtl'))
            ytl = float(box_elem.get('ytl'))
            xbr = float(box_elem.get('xbr'))
            ybr = float(box_elem.get('ybr'))

            # Koordinatları COCO formatına dönüştür (x, y, width, height)
            width = xbr - xtl
            height = ybr - ytl
            bbox = [xtl, ytl, width, height]
            
            area = width * height

            # COCO için anotasyon bilgilerini oluştur
            annotation_info = {
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [],  # Bounding box için boş bırakılır
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco_output['annotations'].append(annotation_info)
            annotation_id_counter += 1

    # Çıktı dizininin var olup olmadığını kontrol et, yoksa oluştur
    output_dir = os.path.dirname(output_json_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Sonuçları JSON dosyasına yaz
    with open(output_json_file, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print("\nConversion complete!")
    print(f"Processed {len(coco_output['images'])} images and {len(coco_output['annotations'])} annotations.")
    print(f"Found {len(coco_output['categories'])} categories: {[cat['name'] for cat in coco_output['categories']]}")
    print(f"COCO JSON file saved to: {output_json_file}")

if __name__ == "__main__":
    # Örnek kullanım
    input_dir = "dataset/mine_deneme/images"
    cvat_xml = "dataset/mine_deneme/annotations.xml"
    output_dir = "dataset/mine_deneme/dataset_tiled
    coco_json = "dataset/mine_deneme/annotation_coco.json"

    convert_cvat_to_coco(cvat_xml, coco_json)

    # 1) Tile et ve COCO dataset üret
    offset_map, categories = cvat_to_coco_tiled(
        input_dir, cvat_xml, output_dir,
        tile_w=300, tile_h=300, stride=300
    )

    # 2) MODELİN TAHMİNLERİNİ BURADA YÜKLE
    # Örn. patch prediction listesi (dummy örnek)
    dummy_preds = [
        {"image_id": 1, "bbox": [50,60,80,90], "score": 0.9, "category_id": 1},
        {"image_id": 2, "bbox": [20,30,60,70], "score": 0.85, "category_id": 1},
    ]

    # 3) Tahminleri stitch et
    final_results = stitch_predictions(dummy_preds, offset_map, iou_thresh=0.5)

    # 4) Yan yana görselleştir
    for fname, res in final_results.items():
        orig_path = glob.glob(os.path.join(input_dir, "**", fname), recursive=True)[0]
        draw_results_side_by_side(orig_path, res,
                                  {v: k for k,v in categories.items()},
                                  save_path=f"{fname}_result.jpg")
```

