import os
import cv2
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/latest.pth'
img_path = 'path/to/your/image.jpg'
save_dir = 'feature_maps_out'

os.makedirs(save_dir, exist_ok=True)

model = init_detector(config_file, checkpoint_file, device='cuda:0')

feature_maps = {}

def get_activation(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

# backbone layers
model.backbone.layer1.register_forward_hook(get_activation('layer1'))
model.backbone.layer2.register_forward_hook(get_activation('layer2'))
model.backbone.layer3.register_forward_hook(get_activation('layer3'))
model.backbone.layer4.register_forward_hook(get_activation('layer4'))

for i, fpn_layer in enumerate(model.neck.lateral_convs):
    fpn_layer.register_forward_hook(get_activation(f'fpn_lateral_{i}'))

with torch.no_grad():
    _ = inference_detector(model, img_path)

img = cv2.imread(img_path)
H, W = img.shape[:2]

for name, fmap in feature_maps.items():
    fmap = fmap[0]  # remove batch dimension 
    fmap_mean = fmap.mean(dim=0).numpy()

    fmap_norm = (fmap_mean - fmap_mean.min()) / (fmap_mean.max() - fmap_mean.min() + 1e-5)
    fmap_norm = (fmap_norm * 255).astype(np.uint8)

    fmap_resized = cv2.resize(fmap_norm, (W, H))

    heatmap = cv2.applyColorMap(fmap_resized, cv2.COLORMAP_JET)

    out_path = os.path.join(save_dir, f'{name}.jpg')
    cv2.imwrite(out_path, heatmap)
    
    """
    for i in range(fmap.shape[0]):
    fmap_ch = fmap[i].numpy()
    fmap_ch = (fmap_ch - fmap_ch.min()) / (fmap_ch.max() - fmap_ch.min() + 1e-5)
    fmap_ch = (fmap_ch * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{name}_ch{i:03d}.jpg'), fmap_ch)
	"""

    print(f"Saved: {out_path}")

print(f"\n Feature maps saved to: {os.path.abspath(save_dir)}")

