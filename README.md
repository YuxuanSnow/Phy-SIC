<h2 align="center">
  <a href="https://yuxuan-xue.com/physic/">PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2510.06219-b31b1b.svg?logo=arXiv)]() 
[![Home Page](https://img.shields.io/badge/Project-Website-C27185.svg)](https://yuxuan-xue.com/physic/) 

[Pradyumna Yalandur Muralidhar](https://pradyumanym.github.io/)<sup>★</sup>,
[Yuxuan Xue](https://yuxuan-xue.com/)<sup>★†</sup>,
[Xianghui Xie](https://virtualhumans.mpi-inf.mpg.de/people/Xie.html),
Margaret Kostyrko,
[Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/)
</h5>

<div align="center">
TL;DR: Human-Scene Interaction and Contact from a Single Image with Physical Plausibility.
</div>
<br>

<div align="center">
    <video width="640" controls>
        <source src="https://yuxuan-xue.com/physic/static/videos/PhySICs_1360_vid.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

## Getting Started

### Installation
Please follow the instructions in [INSTALL.md](INSTALL.md) to set up the environment.

## Downloading Data
Please run the following script to download the required data:
```bash
bash fetch_data.sh
```

This requires access to [SMPL](https://smpl.is.tue.mpg.de/), [SMPL-X](https://smpl-x.is.tue.mpg.de/), [AGORA](https://agora.is.tue.mpg.de/), and [CameraHMR](https://camerahmr.is.tue.mpg.de/). Please enter the credentials when prompted.

### Running the Code
To run the code, please copy images to the `images/` directory. Then, execute the following command:
```bash
python run_optimizer.py
```
This will save the results in the `outputs/` directory.
 
## Roadmap

We maintain a short list of upcoming items and the current release status for key pieces of the project:

- [x] **Demo code release**
- [ ] **Evaluation code**

## Acknowledgements
This code is built on top of many great open-source projects. We would like to thank the authors of the following repositories:
- [OmniEraser](https://github.com/PRIS-CV/Omnieraser) for the image inpainting code.
- [MoGe](https://github.com/microsoft/MoGe) for affine-invariant depth estimation.
- [CameraHMR](https://github.com/pixelite1201/CameraHMR/) for the initial human mesh estimation.
- [WiLoR](https://github.com/rolpotamias/WiLoR)/[WiLoR-mini](https://github.com/warmshao/WiLoR-mini) for initial hand pose estimation.
- [DECO](https://github.com/sha2nkt/deco) and [PROX](https://github.com/mohamedhassanmus/prox) for contact estimation and static contacts.
- [DepthPro](https://github.com/apple/ml-depth-pro) for metric depth estimation.
- [HSfM](https://github.com/hongsukchoi/HSfM_RELEASE) and [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) for the ViTPose inference code and models.
- [LangSAM](https://github.com/luca-medeiros/lang-segment-anything), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [Segment-Anything](https://github.com/facebookresearch/segment-anything) for the segmentation code and models.
- [SMPLFitter](https://github.com/isarandi/smplfitter)/[NLF](https://github.com/isarandi/nlf) for the SMPL-to-SMPL-X converter.

and many others.

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{ym2025physic,
  author    = {Yalandur Muralidhar, Pradyumna and Xue, Yuxuan and Xie, Xianghui and Kostyrko, Margaret and Pons-Moll, Gerard},
  title     = {PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image},
  journal   = {SIGGRAPH Asia 2025 Conference Papers},
  year      = {2025},
}
```
