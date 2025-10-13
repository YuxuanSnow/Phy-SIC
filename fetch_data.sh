#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "Please register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL account): " username
read -p "Password (SMPL account): " password
username=$(urle "${username}")
password=$(urle "${password}")

# SMPL Body model
mkdir -p data/body_models/smpl
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O './data/body_models/smpl/SMPL_python_v.1.1.0.zip' --no-check-certificate --continue
unzip -o ./data/body_models/smpl/SMPL_python_v.1.1.0.zip -d ./data/body_models/smpl/
mv ./data/body_models/smpl/SMPL_python_v.1.1.0/smpl/models/* ./data/body_models/smpl/
rm -r ./data/body_models/smpl/SMPL_python_v.1.1.0 ./data/body_models/smpl/SMPL_python_v.1.1.0.zip
ln -s ./data/body_models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl ./data/body_models/smpl/SMPL_NEUTRAL.pkl

echo -e "Please register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X account): " username
read -p "Password (SMPL-X account): " password
username=$(urle "${username}")
password=$(urle "${password}")

# SMPL-X Body model, part segmentation
mkdir -p data/body_models/smplx
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O './data/body_models/smplx/models_smplx_v1_1.zip' --no-check-certificate --continue
unzip -o ./data/body_models/smplx/models_smplx_v1_1.zip -d ./data/body_models/smplx/
mv ./data/body_models/smplx/models/smplx/* ./data/body_models/smplx/
rm -r ./data/body_models/smplx/models ./data/body_models/smplx/models_smplx_v1_1.zip

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=model_transfer.zip' -O './data/body_models/transfer.zip' --no-check-certificate --continue
unzip ./data/body_models/transfer.zip smpl2smplx_deftrafo_setup.pkl  -d ./data/body_models/
rm ./data/body_models/transfer.zip

echo -e "Please register at https://agora.is.tue.mpg.de"
read -p "Username (CameraHMR account): " username
read -p "Password (CameraHMR account): " password
username=$(urle "${username}")
password=$(urle "${password}")

# Kid templates, required by SMPLFitter
wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=smpl_kid_template.npy" -O "./data/body_models/smpl/kid_template.npy" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=smplx_kid_template.npy" -O "./data/body_models/smplx/kid_template.npy" --no-check-certificate --continue

echo -e "Please register at https://camerahmr.is.tue.mpg.de"
read -p "Username (CameraHMR account): " username
read -p "Password (CameraHMR account): " password
username=$(urle "${username}")
password=$(urle "${password}")

# CameraHMR checkpoints
mkdir -p data/chmr
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=cam_model_cleaned.ckpt' -O './data/chmr/cam_model_cleaned.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=camerahmr_checkpoint_cleaned.ckpt' -O './data/chmr/camerahmr_checkpoint_cleaned.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=model_final_f05665.pkl' -O './data/chmr/model_final_f05665.pkl' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=smpl_mean_params.npz' -O './data/chmr/smpl_mean_params.npz' --no-check-certificate --continue

#ViTPose and Depth Pro
wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-h-wholebody.pth -O data/vitpose_huge_wholebody.pth
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P data/

# Deco checkpoints and data
mkdir -p data/deco
wget https://keeper.mpdl.mpg.de/f/6f2e2258558f46ceb269/?dl=1 --max-redirect=2 --trust-server-names -O data/deco/Release_Checkpoint.tar.gz && tar -xvf data/deco/Release_Checkpoint.tar.gz --directory data/deco && rm -r data/deco/Release_Checkpoint.tar.gz
mv data/deco/Release_Checkpoint/* data/deco/
rmdir data/deco/Release_Checkpoint
wget https://keeper.mpdl.mpg.de/f/50cf65320b824391854b/?dl=1 --max-redirect=2 --trust-server-names -O data/deco/data.tar.gz && tar -xvf data/deco/data.tar.gz --directory data/deco && rm -r data/deco/data.tar.gz
mv data/deco/data/conversions data/
mv data/deco/data/smplx_vert_segmentation.json data/body_models/smplx/
mv data/deco/data/weights/pose_hrnet_w32_256x192.pth data/deco/
mv data/deco/data/smplx/smplx_neutral_tpose.ply data/body_models/smplx/
mv data/deco/data/smpl/smpl_neutral_tpose.ply data/body_models/smpl/
rm -r data/deco/data

# For static contacts 

# echo -e "Please register at https://prox.is.tue.mpg.de"
# read -p "Username (PROX account): " username
# read -p "Password (PROX account): " password
# username=$(urle "${username}")
# password=$(urle "${password}")

# wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=prox&resume=1&sfile=bodysegments.zip" -O "./data/bodysegments.zip" --no-check-certificate --continue
# unzip -o ./data/bodysegments.zip -d ./data/
# rm -r ./data/bodysegments.zip