import argparse
import yaml

# Modüllerin importu
from tile_image import batch_tile_pipeline
from super_resolution import batch_sr_pipeline
from stitch import batch_stitch_pipeline
from split_coco_for_tiles import split_coco_for_tiles


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    config = load_config(args.config)
    if args.tile:
        print(">> Görseller tile'lara ayrılıyor...")
        batch_tile_pipeline(config)
    if args.sr:
        print(">> Super Resolution uygulanıyor...")
        batch_sr_pipeline(config)
    if args.stitch:
        print(">> SR tile'lar birleştiriliyor (stitch)...")
        batch_stitch_pipeline(config)
    print(">> Tüm seçilen adımlar tamamlandı!")
    if args.imagetile:
        print(">> Görsellerin annotation ayırma işlemi yapılıyor...")
        split_coco_for_tiles(config)
        print(">> Annotation ayırma işlemi tamamlandı!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-ESRGAN Script")
    parser.add_argument('--config', type=str, default="config.yaml", help="YAML config dosyasının yolu")
    parser.add_argument('--tile', action='store_true', help="Görselleri tile'a ayır")
    parser.add_argument('--sr', action='store_true', help="Tile'lara Super Resolution uygula")
    parser.add_argument('--stitch', action='store_true', help="SR uygulanmış tile'ları birleştir (stitch)")
    parser.add_argument('--imagetile', action='store_true', help="Görsellerin annotation ayır")

    
    args = parser.parse_args()
    main(args)
    

#python run_pipeline.py --tile

