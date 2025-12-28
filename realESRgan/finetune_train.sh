#!/bin/bash
pip install basicsr facexlib
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop
pip install torch torchvision

# Orijinal checkpoint'i indir
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O models/RealESRGAN_x4plus.pth

# params_ema --> params çevirme (python ile)
python << END
import torch
ckpt = torch.load("models/RealESRGAN_x4plus.pth", map_location='cpu')
torch.save({'params': ckpt['params_ema']}, "models/RealESRGAN_x4plus_FIXED.pth")
END

# functional_tensor uyumluluğu
sed -i 's/functional_tensor/functional/g' /usr/local/lib/python3.*/dist-packages/basicsr/data/degradations.py

# meta_info.txt oluşturma 
python tools/generate_meta_info.py --input_folder ../dataset/tiles --meta_info_path ../dataset/meta_info.txt

python realesrgan/train.py -opt configs/finetune_gan.yml
