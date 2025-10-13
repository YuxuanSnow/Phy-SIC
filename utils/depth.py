import sys
import os
import torch

import numpy as np

import depth_pro

from .utils import (
    align_depth,
    align_depth_pt,
    focal_length_to_intrinsics,
    align_points_pt,
)

sys.path.append(os.path.join(os.getcwd(), "external/Metric3D"))

from moge.model.v1 import MoGeModel

enable_offload = not bool(os.getenv("DISABLE_OFFLOAD"))

# Load the model from huggingface hub (or load from local).
def load_moge(device="cuda"):
    global moge_model
    if "moge_model" in globals() and moge_model is not None:
        pass
    else:
        # moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl")
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
        moge_model.eval()
        if enable_offload:
            moge_model.to("cpu")
        else:
            moge_model.to(device)


def delete_moge():
    global moge_model
    if "moge_model" in globals() and moge_model is not None:
        del moge_model
        moge_model = None


def load_dpro(device="cuda"):
    global dp_model
    global transform
    if "dp_model" in globals() and dp_model is not None:
        pass
    else:
        dp_model, transform = depth_pro.create_model_and_transforms()
        dp_model.eval()
        if enable_offload:
            dp_model.to("cpu")
        else:
            dp_model.to(device)


def delete_dpro():
    global dp_model
    global transform
    if "dp_model" in globals() and dp_model is not None:
        del dp_model
        del transform
        dp_model = None
        transform = None

def run_moge(image, device="cuda"):
    """
    Run MoGe model on the input image.

    Args:
        image (np.array): a numpy image
        device (torch.device): a torch device

    Returns:
        np.array: a numpy array representing the depth map
    """

    if enable_offload:
        moge_model.cuda()
    moge_model.eval()

    # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
    input_image = (
        image  # cv2.cvtColor(cv2.imread("PATH_TO_IMAGE.jpg"), cv2.COLOR_BGR2RGB)
    )
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device)

    if len(input_image.shape) == 3:
        input_image = input_image.permute(2, 0, 1)
    elif len(input_image.shape) == 4:
        input_image = input_image.permute(0, 3, 1, 2)
    else:
        raise ValueError("Input image shape not supported")

    # Infer
    output = moge_model.infer(input_image)
    depth = output["depth"].detach().cpu().numpy()
    pointmap = output["points"].detach().cpu().numpy()
    intrinsics = output["intrinsics"].detach().cpu().numpy()

    if enable_offload:
        moge_model.cpu()

    return depth, pointmap, intrinsics

def run_metric_moge_from_depthpro(image, f_px=None, device="cuda"):
    """
    Get metric depth from MoGe, using alignment with DepthPro.

    Args:
        image (np.array): a numpy image
        device (torch.device): a torch device

    Returns:
        tuple: a tuple of numpy arrays representing the depth map and the camera intrinsics
    """

    depth_moge, _, _ = run_moge(image, device=device)
    depth_metric, f_px = run_depth_pro([image], f_px=f_px, device=device)
    K = focal_length_to_intrinsics(float(f_px), (image.shape[1], image.shape[0]))
    # depth_moge = align_depth(depth_moge, depth_metric)
    depth_moge = align_depth_pt(depth_moge, depth_metric)
    return depth_moge, K

def run_metric_moge_with_human_from_depthpro(
    image, image_human, mask_human, f_px=None, device="cuda", max_depth=50
):
    """
    Get metric depth from MoGe, using alignment with DepthPro.

    Args:
        image (np.array): a numpy image
        image_human (np.array): a numpy image of the human
        mask_human (np.array): a binary mask for the human
        f_px (float, optional): focal length in pixels
        device (torch.device): a torch device
        max_depth (float, optional): maximum depth value

    Returns:
        tuple: a tuple of numpy arrays representing the depth map and the camera intrinsics
    """

    assert image.shape == image_human.shape, (
        "The image and the human image must have the same shape."
    )

    mask_binary = mask_human > 0
    images = np.stack([image, image_human], axis=0)

    depth_moge, points_moge, intrinsics = run_moge(images, device=device)

    depth_moge, depth_human = depth_moge[0], depth_moge[1]
    points_moge, points_human = points_moge[0], points_moge[1]

    # NOTE: This is only required for MoGe v1: v2 returns metric points.
    depth_metric, f_px = run_depth_pro([image], f_px=f_px, device=device)
    K = focal_length_to_intrinsics(float(f_px), (image.shape[1], image.shape[0]))
    points_metric = depth_to_points(depth_metric[None], K=K)
    # use a valid mask to exclude invalid points (as per MoGe)
    mask_invalid = np.isinf(depth_moge)
    # depth_moge = align_depth_pt(depth_moge, depth_metric, mask=~mask_invalid)
    points_moge = align_points_pt(points_moge, points_metric, mask=~mask_invalid)

    mask_binary = mask_binary | (depth_human > max_depth) | (depth_moge > max_depth)
    print(f"mask_binary: {np.sum(mask_binary)}")
    # depth_human = align_depth_pt(depth_human, depth_moge, mask=~mask_binary, iterations=10)
    points_human = align_points_pt(points_human, points_moge, mask=~mask_binary)

    depth_human = np.clip(depth_human, 0, max_depth)
    depth_moge = np.clip(depth_moge, 0, max_depth)

    return depth_moge, depth_human, points_moge, points_human, K


def run_depth_pro(images, f_px=None, device="cuda"):
    """
    Get depth from DepthPro.

    Args:
        images (list): a list of PIL images or numpy arrays
        device (torch.device): a torch device

    Returns:
        np.array: a numpy array representing the depth maps of the input images
        list: a list of focal lengths in pixels
    """
    if enable_offload:
        dp_model.to("cuda")

    dp_model.eval()

    images = [transform(image) for image in images]
    images = torch.stack(images).to(device)

    prediction = dp_model.infer(images, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]

    depth = depth.detach().cpu().numpy()
    if isinstance(focallength_px, torch.Tensor):
        focallength_px = focallength_px.detach().cpu().numpy()

    if enable_offload:
        dp_model.to("cpu")
    return depth, focallength_px


def depth_to_points(depth, K=None, R=None, t=None):
    """
    Reference: https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/utils/geometry.py
    """
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # from reference to target viewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    return pts3D_2[:, :, :, :3, 0][0]
