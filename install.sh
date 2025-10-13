export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True

# create and activate a conda env outside this script
# conda create -n physic python=3.10 -y
# conda activate physic

# Install PyTorch 2.3.1 + torchvision + torchaudio with CUDA 12.1.
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git@918043ed4666eea04da88aa179eb8d27ef4b1a1d
pip install git+https://github.com/jonbarron/robust_loss_pytorch
pip install git+https://github.com/warmshao/WiLoR-mini@a20fc482e68d17c0c8fa19c64f3f4544b6a310cf
pip install git+https://github.com/isarandi/smplfitter.git@13180c45a9201c8113690ad5158fad20b94be36b

cd external/CameraHMR
pip install -r requirements.txt

cd ../
pip install -e ml-depth-pro
pip install ninja==1.11.1.3             # required for fast building of pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

cd ../
pip install -r requirements.txt

pip install -U openmim
pip install git+https://github.com/open-mmlab/mmpose.git@v0.24.0#egg=mmpose
mim install "mmcv==1.3.9" --no-deps
mim install mmdet
cd external
pip install -v -e ViTPose

# This is the final version that works with everything else
pip install numpy==1.26.4