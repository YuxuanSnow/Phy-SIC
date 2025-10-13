### PhySIC: Install Instructions

#### Requirements
This project requires a GPU with atleast 40GB of VRAM. This is due to the OmniEraser model based on FLUX, which is used for image inpainting. 

#### CUDA installation
This project was tested on a CUDA 12.1 toolkit installation. Please download the CUDA 12.1 runfile from [NVIDIA's website](https://developer.nvidia.com/cuda-12-1-0-download-archive), and install the toolkit and set the environment variables as done in [Gen3DSR](https://github.com/AndreeaDogaru/Gen3DSR/blob/main/INSTALL.md).

#### Conda environment setup
To install PhySIC and its dependencies, please create a conda environment with Python 3.10:
```sh
conda create -n physic python=3.10 -y
conda activate physic
```

Finally, run the install script with `bash install.sh`