import os
import argparse
import cv2
import torch
import numpy as np
from smplx import SMPL
from core.utils.renderer_pyrd import Renderer


def make_parser():
    parser = argparse.ArgumentParser(description="CameraHMR dataset visualization")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/training-images",
        help="Path to input image folder.",
    )
    parser.add_argument(
        "--output_folder", type=str, default=".", help="Path to folder output folder."
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="data/training-labels/aic-release.npz",
        help="Path to folder output folder.",
    )
    parser.add_argument("--ind", type=int, default=0, help="index of npz file")
    return parser


def load_smpl_model(model_folder, gender="neutral", num_betas=10):
    return SMPL(
        model_folder,
        model_type="smpl",
        gender=gender,
        ext="npz",
        num_betas=num_betas,
    )


def load_data(npz_path, image_folder, ind):
    data = np.load(npz_path)
    img_path = os.path.join(
        image_folder, data["imgname"][ind].replace("aic-train", "aic-train-vitpose")
    )
    return {
        "img_path": img_path,
        "translations": data["trans_cam"][ind],
        "camera_intrinsics": data["cam_int"][ind],
        "pose": data["pose_cam"][ind],
        "shape": data["shape"][ind],
    }


def render_model(renderer, model_output, img, outdir, file_name_suffix=""):
    front_view = renderer.render_front_view(model_output.vertices, bg_img_rgb=img)
    side_view = renderer.render_side_view(model_output.vertices)
    final_img = np.hstack([img, front_view, side_view])

    overlay_file_name = os.path.join(outdir, f"{file_name_suffix}.png")
    cv2.imwrite(overlay_file_name, final_img)
    print(f"Overlay saved at: {overlay_file_name}")


def main():
    parser = make_parser()
    args = parser.parse_args()
    # Paths and constants
    MODEL_FOLDER = "data/models/SMPL"
    IMAGE_FOLDER = args.image_folder
    NPZ_PATH = args.npz_path
    OUTPUT_DIR = args.output_folder
    ind = args.ind

    # Load SMPL model
    smpl_neutral = load_smpl_model(MODEL_FOLDER)

    # Load data from npz
    data = load_data(NPZ_PATH, IMAGE_FOLDER, ind)

    # Load image
    img = cv2.imread(data["img_path"])
    if img is None:
        raise FileNotFoundError(f"Image not found: {data['img_path']}")
    print(f"Image loaded: {data['img_path']}")

    img_h, img_w, _ = img.shape

    # Extract parameters
    translations = data["translations"]
    camera_intrinsics = data["camera_intrinsics"]
    pose = data["pose"]
    shape = data["shape"]

    # Run SMPL model
    model_output = smpl_neutral(
        betas=torch.tensor(shape).unsqueeze(0).float(),
        global_orient=torch.tensor(pose[:3]).unsqueeze(0).float(),
        body_pose=torch.tensor(pose[3:]).unsqueeze(0).float(),
        transl=torch.tensor(translations).unsqueeze(0),
    )

    # Initialize renderer
    focal_length = camera_intrinsics[0, 0]
    renderer = Renderer(
        focal_length=focal_length,
        img_w=img_w,
        img_h=img_h,
        faces=smpl_neutral.faces,
        same_mesh_color=True,
    )

    # Render and save overlay
    render_model(renderer, model_output, img, OUTPUT_DIR, file_name_suffix="overlay")


if __name__ == "__main__":
    main()
