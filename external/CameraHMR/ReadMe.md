


<div align="center">

# **CameraHMR: Aligning People with Perspective (3DV 2025)**  

[**Priyanka Patel**](https://pixelite1201.github.io/) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [**Michael J. Black**](https://ps.is.mpg.de/person/black)


---

🌐 [**Project Page**](https://camerahmr.is.tue.mpg.de) | 📄 [**ArXiv Paper**](https://arxiv.org/abs/2411.08128) | 🎥 [**Video Results**](https://youtu.be/aDmfAxYLV2w)

---

![](teaser/teaser.png)  
*Figure: CameraHMR Results*

</div>

---


## 🚀 **Release**

- **CameraHMR Demo Code**
- **CameraHMR Training and Evaluation Code**


## 🕒 **Coming Soon**
-  **CamSMPLify Code**  
-  **HumanFoV and DenseKP Code**


## **Installation**
Create a conda environment and install all the requirements.

```
conda create -n camerahmr python=3.10
conda activate camerahmr
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 🎬 **Demo**

###  **Download Required Files**

Download necessary demo files using:

```bash
bash fetch_demo_data.sh
```

Alternatively, download files manually from the [CameraHMR website](https://camerahmr.is.tue.mpg.de). Ensure to update paths in [`constants.py`](core/constants.py) if doing so manually.


Run the demo with following command. It will run demo on all images in the specified --image_folder, and save renderings of the reconstructions and the output mesh in --out_folder.

```
python demo.py --image_folder demo_images --output_folder output_images
```

##  **4DHumans Labels with full perspective camera**

You can download the training data; fitted SMPL parameters for **INSTA/AIC/COCO/MPII images** from the [CameraHMR website](https://camerahmr.is.tue.mpg.de/index.html) (registration required).

Alternatively, use the following script:

```bash
bash scripts/fetch_4dhumans_training_labels.sh
```

> **Note:** We cannot provide the original AIC/INSTA images. These images must be obtained from their original sources. For convenience, you can use the [4D-Humans repository](https://github.com/shubham-goel/4D-Humans?tab=readme-ov-file), which offers these images in WebDataset format. To extract images from the WebDataset, refer to [this script](core/utils/extract_images_from4dhumans.py).


To overlay the fitted SMPL mesh on your images, use the following command:

```bash
python dataset_vis.py --image_folder path_to_img_folder --output_folder path_for_output_file --npz_path path_to_npz_file
```
path_to_img_folder corresponds to path of download INSTA/AIC images. path_to_npz_file corresponds to downloaded SMPL params.


## Training and Evaluation
Please check out the [document](docs/training.md) for Training and Evaluation detail instructions. 

## 🙌 **Acknowledgements**

This project leverages outstanding resources from:

- [ 4D-Humans](https://github.com/shubham-goel/4D-Humans?tab=readme-ov-file)  
- [ BEDLAM](https://bedlam.is.tue.mpg.de/)  
- [ SMPLify](https://smplify.is.tue.mpg.de/)  
- [ ViTPose](https://github.com/ViTAE-Transformer/ViTPose)  
- [ Detectron2](https://github.com/facebookresearch/detectron2)


## 📚 **Citation**

If you find **CameraHMR** useful in your work, please cite:

```bibtex
@article{patel2024camerahmr,
  title={CameraHMR: Aligning People with Perspective},
  author={Patel, Priyanka and Black, Michael J},
  journal={arXiv preprint arXiv:2411.08128},
  year={2024}
}
```



<div align="center">

✨ **Thank you for your interest in CameraHMR!** ✨

</div>

