import sys
import json
import math
from typing import Any, Dict, List, Tuple


def flatten_segmentation(seg: Any) -> List[float]:
    if isinstance(seg, list):
        if len(seg) > 0 and isinstance(seg[0], list):
            # polygon with multiple parts
            return [float(v) for part in seg for v in part]
        return [float(v) for v in seg]
    # RLE or unsupported
    return []


def validate_coco(path: str) -> None:
    data = json.load(open(path, "r"))
    images = data.get("images", [])
    if not images:
        print("No images found in JSON")
        return
    if len(images) > 1:
        print(f"Warning: {len(images)} images present; validating against the first.")
    img = images[0]
    width = img.get("width")
    height = img.get("height")
    anns = data.get("annotations", [])

    total = 0
    bad_even = 0
    bad_neg = 0
    bad_nan = 0
    bad_out = 0
    empty_seg = 0
    maxx = float("-inf")
    maxy = float("-inf")
    minx = float("inf")
    miny = float("inf")

    for ann in anns:
        seg = ann.get("segmentation", [])
        flat = flatten_segmentation(seg)
        if not flat:
            empty_seg += 1
            continue
        total += 1
        if len(flat) % 2 != 0 or len(flat) < 6:
            bad_even += 1
        for i, v in enumerate(flat):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                bad_nan += 1
            if i % 2 == 0:
                x = v
                minx = min(minx, x)
                maxx = max(maxx, x)
                if x < 0:
                    bad_neg += 1
                if isinstance(width, (int, float)) and x > float(width):
                    bad_out += 1
            else:
                y = v
                miny = min(miny, y)
                maxy = max(maxy, y)
                if y < 0:
                    bad_neg += 1
                if isinstance(height, (int, float)) and y > float(height):
                    bad_out += 1

    print(f"image={img.get('file_name')} size={width}x{height}")
    print(f"annotations_total={len(anns)} with_polygons={total} empty_or_rle={empty_seg}")
    print(f"issues: bad_even={bad_even} bad_neg={bad_neg} bad_out_of_bounds={bad_out} bad_nan={bad_nan}")
    if total:
        print(f"coord_min=({minx:.3f},{miny:.3f}) coord_max=({maxx:.3f},{maxy:.3f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_segmentation.py <path_to_coco_or_per_image_json>")
        sys.exit(1)
    validate_coco(sys.argv[1])


