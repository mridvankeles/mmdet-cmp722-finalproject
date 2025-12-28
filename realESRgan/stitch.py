import os
import re
import numpy as np
from PIL import Image
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def natural_sort_key(s): 
    return [int(text) if text.isdigit() else text.lower()  
            for text in re.split('(\d+)', s)]

def auto_detect_grid(sr_tiles_dir, stitch_ext): 
    files = sorted([f for f in os.listdir(sr_tiles_dir) if f.endswith(stitch_ext)], key=natural_sort_key) 
    num_tiles = len(files) 
    num_cols = int(np.ceil(np.sqrt(num_tiles)))
    num_rows = int(np.ceil(num_tiles / num_cols))
    return num_rows, num_cols, files

def stitch_tiles_fixed_grid(sr_tiles_dir, output_path, grid_shape=None, stitch_ext=None):
    # config'ten çekme işlemi 
    if stitch_ext is None:
        stitch_ext = config.get("stitch_ext", ".png")
        # config'ten gelen uzantı başında nokta yoksa ekle
        if not stitch_ext.startswith("."):
            stitch_ext = "." + stitch_ext

    files = sorted([f for f in os.listdir(sr_tiles_dir) if f.endswith(stitch_ext)], key=natural_sort_key) #klasördeki tüm dosyaları alıp sıralama 
    if not files:
        print(f"No tiles found in {sr_tiles_dir} with extension {stitch_ext}")
        return
    if grid_shape is None:     # grid_shape config'ten gelmiyorsa otomatik olarak hesaplama 
        num_rows, num_cols, _ = auto_detect_grid(sr_tiles_dir, stitch_ext)
    else:
        num_rows, num_cols = grid_shape
    assert len(files) == num_rows * num_cols, f"{len(files)} tile var ama grid {num_rows}x{num_cols}={num_rows*num_cols}!"   # grid ile tile sayısı uyuşuyor mu kontrol edilir
    sample_tile = np.array(Image.open(os.path.join(sr_tiles_dir, files[0])))    #İlk tile’ı açar ve yüksekliğini (h), genişliğini (w), kanal sayısını (R/G/B veya gri) alır.
    tile_h, tile_w = sample_tile.shape[:2] 
    channels = 1 if sample_tile.ndim == 2 else sample_tile.shape[2]  
    stitched = np.zeros((num_rows * tile_h, num_cols * tile_w, channels), dtype=sample_tile.dtype)  #tile’ların birleşeceği büyüklükte (ve aynı tipte) boş bir numpy array oluşturur
    for idx, fname in enumerate(files):   #tile’ları sırayla okur ve stitched array’e yerleştirir
        row = idx // num_cols
        col = idx % num_cols
        tile = np.array(Image.open(os.path.join(sr_tiles_dir, fname))) 
        if tile.ndim == 2 and channels == 3:  
            tile = np.stack([tile]*3, axis=-1) 
        stitched[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w, ...] = tile 
        
    if stitched.shape[2] == 1:
        stitched = stitched.squeeze(-1)

    # Convert to PIL for saving and resizing
    stitched_img = Image.fromarray(stitched.astype(np.uint8))

    # --------------------------------------------------------------
    # 🚀 NEW: Optional high-quality downscale (LANCZOS)
    # --------------------------------------------------------------
    
    print(f"  >> Downscaling stitched image to {800}x{800} using LANCZOS")
    stitched_img = stitched_img.resize((800, 800), Image.Resampling.LANCZOS)
    
    stitched_img.save(output_path)
    print(f"Stitched image saved: {output_path}")
    return stitched_img.size

def batch_stitch_pipeline(config):
    import os

    sr_tiles_dir = config["sr_tiles_dir"]
    stitched_dir = config["stitched_dir"]
    os.makedirs(stitched_dir, exist_ok=True)
    stitch_ext = config.get("stitch_ext", ".png")
    if not stitch_ext.startswith("."):
        stitch_ext = "." + stitch_ext

    # alt klasörleri dolaş
    for folder_name in os.listdir(sr_tiles_dir):
        folder_path = os.path.join(sr_tiles_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Dosya ise atla
        # Stitch işlemi her alt klasördeki tile'lar için yapılır
        stitched_outfile = os.path.join(stitched_dir, folder_name + stitch_ext)
        print(f">> Birleştiriliyor: {folder_name} -> {stitched_outfile}")
        result_shape = stitch_tiles_fixed_grid(
            folder_path,        # sr_tiles_dir yerine alt klasör!
            stitched_outfile,
            stitch_ext=stitch_ext
        )
        if result_shape:
            print(f"Birleştirme tamamlandı! Boyut: {result_shape}")
        else:
            print(f"Birleştirme başarısız oldu: {folder_name}")

    print(">> Tüm stitching işlemleri tamamlandı!")
