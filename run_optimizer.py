import os
import pickle
import glob
import time
import subprocess
import numpy as np

from pathlib import Path

from omegaconf import OmegaConf
import torch
from utils.vis import get_scene

def get_gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Get the first GPU's memory in MB
            memory_mb = int(result.stdout.strip().split("\n")[0])
            return memory_mb / 1024  # Convert to GB
    except:
        pass
    return 0


# Only set the environment variable if GPU has more than 60GB of memory
if get_gpu_memory() > 60:
    print(
        f"Setting DISABLE_OFFLOAD to 1 due to sufficient GPU memory. Available: {get_gpu_memory()} GB"
    )
    os.environ["DISABLE_OFFLOAD"] = "1"

# Load config directly from YAML instead of using Hydra
cfg_path = Path("cfg") / "v1.yaml"
if not cfg_path.exists():
    raise FileNotFoundError(f"Config file not found: {cfg_path}")
cfg = OmegaConf.load(str(cfg_path))
print(OmegaConf.to_yaml(cfg))

from optimizer import (
    HumanScene,
    load_gsam,
    load_omni,
    load_chmr,
    load_deco,
    load_wilor,
    load_moge,
    load_dpro,
    load_vitpose,
)

t0 = time.time()
load_gsam()
print("Time taken to load gsam: {:.4f} seconds".format(time.time() - t0))

t0 = time.time()
load_vitpose()
print("Time taken to load vitpose: {:.4f} seconds".format(time.time() - t0))

t0 = time.time()
load_omni()
print("Time taken to load sdxl: {:.4f} seconds".format(time.time() - t0))

t0 = time.time()
if cfg.smpl_model == "chmr":
    load_chmr()
else:
    raise ValueError("Unknown SMPL model: {}".format(cfg.smpl_model))
print("Time taken to load chmr: {:.4f} seconds".format(time.time() - t0))

t0 = time.time()
load_deco()
print("Time taken to load deco: {:.4f} seconds".format(time.time() - t0))

t0 = time.time()
load_wilor()
print("Time taken to load wilor: {:.4f} seconds".format(time.time() - t0))

t0 = time.time()
load_moge()
print("Time taken to load moge: {:.4f} seconds".format(time.time() - t0))

t0 = time.time()
load_dpro()
print("Time taken to load dpro: {:.4f} seconds".format(time.time() - t0))

output_foldername = cfg.run_name
image_paths = list(glob.glob("./images/*"))

failed_images = []
for image_path in image_paths:
    print("Processing image: ", image_path)

    img_path = Path(image_path)
    file_name = img_path.stem
    out_dir = Path("outputs") / output_foldername / file_name

    # try:
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        x = HumanScene(cfg, image_path=image_path, output_path=out_dir)
    # except Exception as e:
    #     print(f"Error in processing {image_path}: {e}")
    #     failed_images.append(image_path)
    #     continue

    # Save numpy arrays.
    data = {
        "depth": x.depth.cpu().numpy(),
        "K": x.K.cpu().numpy(),
        "pts3d": x.pts3d.cpu().numpy(),
        "inlier_mask": x.inlier_mask.cpu().numpy(),
        "scale": x.scale.detach().cpu().numpy(),
        "normals": x.normals.cpu().numpy(),
        "plane_points": x.plane_points.cpu().numpy()
        if hasattr(x.plane_points, "cpu")
        else x.plane_points,
        "plane_normal": x.plane_normal.cpu().numpy()
        if hasattr(x.plane_normal, "cpu")
        else x.plane_normal,
        "body_params": {k: v.detach().cpu().numpy() for k, v in x.body_params.items()},
        "cam_trans": x.cam_trans.detach().cpu().numpy(),
    }

    with open(x.outpath / "scene_data_final.pkl", "wb") as f:
        pickle.dump(data, f)

    with torch.amp.autocast(enabled=False, device_type="cuda"):
        scene = get_scene(out_dir, max_faces=int(1e18))
    scene.export(os.path.join(out_dir, "humanscene.ply"))

    
print("Failed images: ", failed_images)
