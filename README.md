# TODbox (Tiny Object Detection Box)

**We have now released the full sets (trainval, test) of AI-TOD-v2!** [[Download]](https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing)

This is a repository of the official implementation of the following paper:

- [[Paper]](https://www.sciencedirect.com/science/article/pii/S0924271622001599?dgcid=author)[[Code]](mmdet-nwdrka) Detecting tiny Objects in aerial images: A normalized Wasserstein distance and A new benchmark ([ISPRS J P & RS](https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing), 2022)
- [[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Xu_Dot_Distance_for_Tiny_Object_Detection_in_Aerial_Images_CVPRW_2021_paper.html)[[Code]](mmdet-nwdrka) Dot distance for tiny object detection in aerial images ([CVPRW](http://www.classic.grss-ieee.org/earthvision2021/), 2021)

## Introduction

The Normalized Wasserstein Distance and the RanKing-based Assigning strategy (NWD-RKA) for tiny object detection.
![demo image](figures/nwdrka.PNG)

A comparison between AI-TOD and AI-TOD-v2.
![demo image](figures/fps2.gif)

## Supported Data

- [x] [AI-TOD](https://github.com/jwwangchn/AI-TOD)
- [x] [AI-TOD-v2](https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing)

Notes: The images of the **AI-TOD-v2** are the same of the **AI-TOD**. In this stage, we only release the train, val annotations of the **AI-TOD-v2**, the test annotations will be used to hold further competitions.

## Supported Methods

Supported baselines for tiny object detection:

- [x] [Baselines](mmdet-nwdrka/configs_nwdrka/baseline)

Supported horizontal tiny object detection methods:

- [x] [DotD](mmdet-nwdrka/configs_nwdrka/nwd_rka)
- [x] [NWD-RKA](mmdet-nwdrka/configs_nwdrka/nwd_rka)
- [ ] [RFLA](https://github.com/Chasel-Tsui/mmdet-rfla)

Supported rotated tiny object detection methods:

- [ ] [DCFL](https://github.com/Chasel-Tsui/mmrotate-dcfl)

## Installation and Get Started

Required environments:

- Linux
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)

Install TODbox:

Note that our TODbox is based on the [MMDetection 2.24.1](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/Chasel-Tsui/mmdet-aitod.git
cd mmdet-nwdrka
pip install -r requirements/build.txt
python setup.py develop
```

## Citation

If you use this repo in your research, please consider citing these papers.

```
@inproceedings{xu2021dot,
  title={Dot Distance for Tiny Object Detection in Aerial Images},
  author={Xu, Chang and Wang, Jinwang and Yang, Wen and Yu, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={1192--1201},
  year={2021}
}

@inproceedings{NWDRKA_2022_ISPRS,
    title={Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and A New Benchmark},
    author={Xu, Chang and Wang, Jinwang and Yang, Wen and Yu, Huai and Yu, Lei and Xia, Gui-Song},
    booktitle={ISPRS Journal of Photogrammetry and Remote Sensing},
    volume={190},
    pages={79--93},
    year={2022},
}
```

## References

- [AI-TOD](https://github.com/jwwangchn/AI-TOD)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [DOTA](https://captain-whu.github.io/DOTA/index.html)

---

## CMP722 Advanced Vision Project: Super-Resolution and Attention Mechanisms for Tiny Object Detection

**Authors:** Muhammed Rıdvan Keleş & Refik Can Öztaş

This work was conducted as a final project for the MS course **CMP722 Advanced Vision**. This repository contains enhancements to the AI-TOD dataset and detection models, specifically focusing on super-resolution and attention mechanisms.

### 1. Super-Resolution with Real-ESRGAN

We integrated **Real-ESRGAN** to enhance the AI-TOD dataset.

- **Training**: We fine-tuned Real-ESRGAN (4x) using the **DOTA remote sensing dataset**.
- **Enhancement**: The fine-tuned model was used to super-resolve the AI-TOD dataset images.
- **Validation**: We confirmed the effectiveness of this enhancement in improving image quality for tiny objects.

### 2. Model Enhancements (CSAM Plugin)

We enhanced the **ResNet-50** backbone by integrating the **CSAM (Channel-Spatial Attention Module)** plugin.

- **Comparison**: We conducted comparative experiments between the standard ResNet-50 and the CSAM-enhanced version (with vs. without attention).

### 3. Method Comparison

We compared the performance of two ranking strategies:

- **NWDRKA** (Normalized Wasserstein Distance with Ranking-based Assigning)
- **IoU Ranking**

### 4. Environment & Implementation

- **Super-Resolution Pipeline**: Implemented and tested locally.
- **Model Training & Evaluation**: Executed on **Google Colab**.
