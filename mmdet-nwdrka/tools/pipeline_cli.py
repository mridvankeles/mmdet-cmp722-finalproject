import argparse
import json
import os
import sys
from typing import Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from autolabel import (
    tile_unannotated_image_folder,
    run_mmdet_inference_on_folder,
    convert_to_coco_list,
    get_all_original_image_dimensions,
    stitch_predictions as stitch_with_clip,
    autolabel_predictions_to_coco_json,
    tile_coco_annotations
)


def read_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)


def cmd_tile(cfg: Dict[str, Any]) -> None:
    input_image_folder = cfg["input_image_folder"]
    tile_cfg = cfg["tile"]
    tile_output_dir = tile_cfg.get("tile_output_dir", os.path.join(cfg.get("output_root", "workdir/autolabel_run"), "tiles"))

    os.makedirs(tile_output_dir, exist_ok=True)

    offset_map, tiles_meta = tile_unannotated_image_folder(
        input_image_folder,
        tile_output_dir,
        tile_cfg["width"],
        tile_cfg["height"],
        tile_cfg["stride"],
    )

    # cache for later steps
    cache = cfg.get("cache", {})
    if "offset_map_json" in cache:
        os.makedirs(os.path.dirname(cache["offset_map_json"]), exist_ok=True)
        with open(cache["offset_map_json"], 'w') as f:
            json.dump({str(k): v for k, v in offset_map.items()}, f)
    if "tiles_metadata_json" in cache:
        os.makedirs(os.path.dirname(cache["tiles_metadata_json"]), exist_ok=True)
        with open(cache["tiles_metadata_json"], 'w') as f:
            json.dump(tiles_meta, f)
    return offset_map,tiles_meta

def cmd_infer(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg["model"]
    tile_cfg = cfg["tile"]
    tile_output_dir = tile_cfg.get("tile_output_dir", os.path.join(cfg.get("output_root", "workdir/autolabel_run"), "tiles"))
    inference_cfg = cfg["inference"]

    results = run_mmdet_inference_on_folder(
        model_cfg["config_file"],
        model_cfg["checkpoint_file"],
        os.path.join(tile_output_dir, "images"),
        inference_cfg["pickle_path"],
        device=model_cfg.get("device", "cuda:0"),
        cfg_options=model_cfg.get("cfg_options", None),
    )
    print(f"Saved predictions pickle to {inference_cfg['pickle_path']} with {len(results)} images.")


def cmd_stitch(cfg: Dict[str, Any]) -> None:
    inference_cfg = cfg["inference"]
    with open(inference_cfg["pickle_path"], 'rb') as f:
        import pickle
        raw_predictions = pickle.load(f)

    tiled_prediction_list = convert_to_coco_list(raw_predictions)

    cache = cfg.get("cache", {})
    if os.path.exists(cache.get("offset_map_json", "")):
        with open(cache["offset_map_json"], 'r') as f:
            offset_map = {int(k): tuple(v) for k, v in json.load(f).items()}
    else:
        tiles_dir = cfg["tile"].get("tile_output_dir")
        if not tiles_dir:
            raise RuntimeError("offset_map_json missing and tile_output_dir not provided to rebuild.")
        raise RuntimeError("offset_map_json is required. Run 'tile' first.")

    # dimension map
    input_image_folder = cfg["input_image_folder"]
    if os.path.exists(cache.get("dimensions_json", "")):
        with open(cache["dimensions_json"], 'r') as f:
            dimension_map = {k: tuple(v) for k, v in json.load(f).items()}
    else:
        dimension_map = get_all_original_image_dimensions(input_image_folder)
        if "dimensions_json" in cache:
            os.makedirs(os.path.dirname(cache["dimensions_json"]), exist_ok=True)
            with open(cache["dimensions_json"], 'w') as f:
                json.dump({k: [w, h] for k, (w, h) in dimension_map.items()}, f)

    # Stitch with clipping and class-aware NMS
    iou_thresh = cfg.get("stitch", {}).get("iou_thresh")
    final_preds = stitch_with_clip(
        tiled_prediction_list,
        offset_map,
        dimension_map,
        iou_thresh=iou_thresh,
    )

    stitched_json = cfg.get("stitch", {}).get("stitched_json")
    if stitched_json:
        os.makedirs(os.path.dirname(stitched_json), exist_ok=True)
        with open(stitched_json, 'w') as f:
            json.dump(final_preds, f)
        print(f"Saved stitched predictions JSON to {stitched_json}")


def cmd_export_coco(cfg: Dict[str, Any]) -> None:
    # Load stitched predictions JSON
    stitched_json = cfg.get("stitch", {}).get("stitched_json")
    if not stitched_json or not os.path.exists(stitched_json):
        raise RuntimeError("Stitched predictions JSON not found. Run 'stitch' first.")
    with open(stitched_json, 'r') as f:
        final_preds = json.load(f)

    categories_raw = cfg.get("categories", {})
    categories = {int(k): v for k, v in categories_raw.items()}

    coco_output_json = cfg.get("export", {}).get("coco_output_json")
    if not coco_output_json:
        raise RuntimeError("export.coco_output_json not set in config")

    pred_coco =autolabel_predictions_to_coco_json(
        final_preds=final_preds,
        category_label_map=categories,
        original_image_folder=cfg["input_image_folder"],
        output_json_path=coco_output_json,
    )
    return pred_coco

def cmd_tile_annot(cfg):
    
    coco_json_data = tile_coco_annotations(
        cfg.get("export", {}).get("coco_output_json"), 
        cfg.get("cache",{}).get("offset_map_json"), 
        cfg.get("cache", {}).get("tiles_metadata_json"),
        cfg.get("export", {}).get("tiled_coco_output_json")
    )
    return coco_json_data

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference tiling/stitching/COCO CLI")
    parser.add_argument("command", choices=["tile-image","infer", "stitch", "export-coco","tile-annot", "all"], help="Pipeline step to run")
    parser.add_argument("--config", dest="config", default="inference_config.json", help="Path to JSON config file")

    args = parser.parse_args()
    cfg = read_config(args.config)

    if args.command == "tile-image":
        offset_map,tiled_metadata=cmd_tile(cfg)
    elif args.command == "tile-annot":
        coco_json_data = cmd_tile_annot(cfg)
    elif args.command == "infer":
        cmd_infer(cfg)
    elif args.command == "stitch":
        cmd_stitch(cfg)
    elif args.command == "export-coco":
        cmd_export_coco(cfg)
    elif args.command == "all":
        cmd_tile(cfg)
        cmd_infer(cfg)
        cmd_stitch(cfg)
        coco_annot=cmd_export_coco(cfg)
        cmd_tile_annot(cfg)

        for ann in coco_annot["annotations"]:
            x,y,w,h = ann["bbox"]
            x_center = x+w/2
            y_center = y+h/2
            print(f"x_center: {x_center}, y_center: {y_center}")





if __name__ == "__main__":
    main()
