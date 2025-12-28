import os
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def prepare_sr_model(model_path, device="cuda", scale=4):
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)  
    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        half=False,
        device=device
    )

def batch_sr_pipeline(config):
    sr_model = prepare_sr_model(
        model_path=config["model_path"],
        device=config.get("device", "cuda"),
        scale=config["scale"]
    )

    for folder in tqdm(os.listdir(config["tiles_dir"])):
        img_dir = os.path.join(config["tiles_dir"], folder)
        out_dir = os.path.join(config["sr_tiles_dir"], folder)
        os.makedirs(out_dir, exist_ok=True)
        tile_files = [f for f in os.listdir(img_dir) if f.endswith(config["output_format"])]
        for tile_file in tile_files:
            img_tile_path = os.path.join(img_dir, tile_file)
            if os.path.exists(os.path.join(out_dir, tile_file)) == False:
                img_tile = np.array(Image.open(img_tile_path))
                # Gri ise 3 kanala çevir
                if img_tile.ndim == 2:
                    img_tile = np.stack([img_tile]*3, axis=-1)
                sr_output, _ = sr_model.enhance(img_tile, outscale=config["scale"])
                Image.fromarray(sr_output.astype(np.uint8)).save(os.path.join(out_dir, tile_file))
            else:
                print(f"{img_tile_path} already exists.")
        #print(f"SR uygulandı: {folder}")

