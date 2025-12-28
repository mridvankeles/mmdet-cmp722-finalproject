import os
import numpy as np
from PIL import Image, ImageFile
import warnings
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Allow very large images (e.g., big GeoTIFF) without triggering Pillow's DecompressionBombError
# This is safe if you trust your input images and have enough RAM/disk.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

def _to_uint8_image_array(img: np.ndarray) -> np.ndarray:
    """Convert any image array to uint8 for safe PNG saving.

    - uint8: returned as-is
    - uint16: downscale by 256 (>> 8)
    - float: assume 0..1 or 0..255, normalize to 0..255 then cast
    - other integer types: clip to 0..255 then cast
    """
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    if np.issubdtype(img.dtype, np.floating):
        # Try to detect range; if max <= 1.0, scale to 0..255; else clip to 0..255
        max_val = np.nanmax(img) if img.size > 0 else 1.0
        if max_val <= 1.0:
            scaled = (img * 255.0)
        else:
            scaled = np.clip(img, 0.0, 255.0)
        return scaled.astype(np.uint8)
    # Other integer types
    return np.clip(img, 0, 255).astype(np.uint8)


def tile_image(image_path, out_dir, tile_size, stride, output_format):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_tile_dir = os.path.join(out_dir, base_name)
    os.makedirs(image_tile_dir, exist_ok=True)
    try:
        pil_img = Image.open(image_path)
        # Enforce a consistent read mode unless overridden
        read_mode = config.get("tile_read_mode", "RGB")  # options like "RGB", "L"
        if read_mode:
            try:
                pil_img = pil_img.convert(read_mode)
            except Exception:
                pass
        img = np.array(pil_img)
        # If more than 3 channels, keep first 3 to avoid unexpected bands
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        # Ensure uint8 for stable PNG saving unless explicitly kept
        if not config.get("keep_source_bitdepth", False):
            img = _to_uint8_image_array(img)
    except Exception as e:
        raise RuntimeError(f"Failed to open image for tiling: {image_path}. Error: {e}")
    height, width = img.shape[:2]
    tile_id = 1
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            img_tile = img[y:y_end, x:x_end]
            #Dinamik padding
            if img_tile.shape[0] < tile_size or img_tile.shape[1] < tile_size: # tile boyutundan küçüks
                if img.ndim == 2: # gri resim ise
                    padded = np.zeros((tile_size, tile_size), dtype=img.dtype)  
                    padded[:img_tile.shape[0], :img_tile.shape[1]] = img_tile
                else: 
                    padded = np.zeros((tile_size, tile_size, img.shape[2]), dtype=img.dtype) 
                    padded[:img_tile.shape[0], :img_tile.shape[1], :] = img_tile
                img_tile = padded
            tile_filename = f"tile_{tile_id:05d}.{output_format}" 
            Image.fromarray(img_tile).save(os.path.join(image_tile_dir, tile_filename))
            tile_id += 1

def batch_tile_pipeline(config):
    image_files = [f for f in os.listdir(config["images_dir"]) if f.endswith(config["image_suffix"])]
    for img_name in image_files:
        image_path = os.path.join(config["images_dir"], img_name)
        tile_image(image_path, config["tiles_dir"], config["tile_size"], config["stride"], config["output_format"])
        print(f"Tiling işlemi tamamlandı: {img_name}")
