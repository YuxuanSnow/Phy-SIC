#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }



# CameraHMR
echo -e "\nYou need to register at https://camerahmr.is.tue.mpg.de/"
read -p "Username (CameraHMR):" username
read -p "Password (CameraHMR):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/test-labels
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=test-labels.zip' -O './data/test-labels.zip' --no-check-certificate --continue
unzip data/test-labels.zip -d data/

mkdir -p data/models/SMPL
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=SMPL.zip' -O './data/models/SMPL.zip' --no-check-certificate --continue
unzip data/models/SMPL.zip -d data/models/

mkdir -p data/models/SMPLX
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=SMPLX.zip' -O './data/models/SMPLX.zip' --no-check-certificate --continue
unzip data/models/SMPLX.zip -d data/models/

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=train-eval-utils.zip' -O './data/train-eval-utils.zip' --no-check-certificate --continue
unzip data/train-eval-utils.zip -d data/
