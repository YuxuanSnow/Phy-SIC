import os
import sys
import trimesh
import torch
import cv2
import smplx
import pickle
import json
from PIL import Image
from multiprocessing import Pool
from typing import List

sys.path = [
    ".",
    "external/CameraHMR",
] + sys.path

os.environ["DATA_ROOT"] = "data"

import numpy as np
import open3d as o3d
from torchvision.transforms import Normalize
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)
from smplfitter.pt import BodyConverter, BodyModel
from sklearn.neighbors import KDTree
from pytorch3d.ops import utils as oputil
from pytorch3d.ops import knn_points
from pytorch3d.structures.pointclouds import Pointclouds

from models.deco.deco import DECO
from core.datasets.dataset import Dataset
from core.utils import recursive_to
from mesh_estimator import HumanMeshEstimator
from models.hsfm.joint_names import COCO_WHOLEBODY_KEYPOINTS, ORIGINAL_SMPLX_JOINT_NAMES
from utils.geometry import aa_to_rotmat, rotmat_to_aa
from utils.utils import find_closest_boxes_by_iou
from utils.constants import part_segments

# Taken from sapiens (https://github.com/facebookresearch/sapiens)
COCO_WHOLEBODY_SKELETON_INFO = {
    0: dict(link=(15, 13), id=0, color=[0, 255, 0]),
    1: dict(link=(13, 11), id=1, color=[0, 255, 0]),
    2: dict(link=(16, 14), id=2, color=[255, 128, 0]),
    3: dict(link=(14, 12), id=3, color=[255, 128, 0]),
    4: dict(link=(11, 12), id=4, color=[51, 153, 255]),
    5: dict(link=(5, 11), id=5, color=[51, 153, 255]),
    6: dict(link=(6, 12), id=6, color=[51, 153, 255]),
    7: dict(link=(5, 6), id=7, color=[51, 153, 255]),
    8: dict(link=(5, 7), id=8, color=[0, 255, 0]),
    9: dict(link=(6, 8), id=9, color=[255, 128, 0]),
    10: dict(link=(7, 9), id=10, color=[0, 255, 0]),
    11: dict(link=(8, 10), id=11, color=[255, 128, 0]),
    12: dict(link=(1, 2), id=12, color=[51, 153, 255]),
    13: dict(link=(0, 1), id=13, color=[51, 153, 255]),
    14: dict(link=(0, 2), id=14, color=[51, 153, 255]),
    15: dict(link=(1, 3), id=15, color=[51, 153, 255]),
    16: dict(link=(2, 4), id=16, color=[51, 153, 255]),
    17: dict(link=(3, 5), id=17, color=[51, 153, 255]),
    18: dict(link=(4, 6), id=18, color=[51, 153, 255]),
    19: dict(link=(15, 17), id=19, color=[0, 255, 0]),
    20: dict(link=(15, 18), id=20, color=[0, 255, 0]),
    21: dict(link=(15, 19), id=21, color=[0, 255, 0]),
    22: dict(link=(16, 20), id=22, color=[255, 128, 0]),
    23: dict(link=(16, 21), id=23, color=[255, 128, 0]),
    24: dict(link=(16, 22), id=24, color=[255, 128, 0]),
    25: dict(link=(91, 92), id=25, color=[255, 128, 0]),
    26: dict(link=(92, 93), id=26, color=[255, 128, 0]),
    27: dict(link=(93, 94), id=27, color=[255, 128, 0]),
    28: dict(link=(94, 95), id=28, color=[255, 128, 0]),
    29: dict(link=(91, 96), id=29, color=[255, 153, 255]),
    30: dict(link=(96, 97), id=30, color=[255, 153, 255]),
    31: dict(link=(97, 98), id=31, color=[255, 153, 255]),
    32: dict(link=(98, 99), id=32, color=[255, 153, 255]),
    33: dict(link=(91, 100), id=33, color=[102, 178, 255]),
    34: dict(link=(100, 101), id=34, color=[102, 178, 255]),
    35: dict(link=(101, 102), id=35, color=[102, 178, 255]),
    36: dict(link=(102, 103), id=36, color=[102, 178, 255]),
    37: dict(link=(91, 104), id=37, color=[255, 51, 51]),
    38: dict(link=(104, 105), id=38, color=[255, 51, 51]),
    39: dict(link=(105, 106), id=39, color=[255, 51, 51]),
    40: dict(link=(106, 107), id=40, color=[255, 51, 51]),
    41: dict(link=(91, 108), id=41, color=[0, 255, 0]),
    42: dict(link=(108, 109), id=42, color=[0, 255, 0]),
    43: dict(link=(109, 110), id=43, color=[0, 255, 0]),
    44: dict(link=(110, 111), id=44, color=[0, 255, 0]),
    45: dict(link=(112, 113), id=45, color=[255, 128, 0]),
    46: dict(link=(113, 114), id=46, color=[255, 128, 0]),
    47: dict(link=(114, 115), id=47, color=[255, 128, 0]),
    48: dict(link=(115, 116), id=48, color=[255, 128, 0]),
    49: dict(link=(112, 117), id=49, color=[255, 153, 255]),
    50: dict(link=(117, 118), id=50, color=[255, 153, 255]),
    51: dict(link=(118, 119), id=51, color=[255, 153, 255]),
    52: dict(link=(119, 120), id=52, color=[255, 153, 255]),
    53: dict(link=(112, 121), id=53, color=[102, 178, 255]),
    54: dict(link=(121, 122), id=54, color=[102, 178, 255]),
    55: dict(link=(122, 123), id=55, color=[102, 178, 255]),
    56: dict(link=(123, 124), id=56, color=[102, 178, 255]),
    57: dict(link=(112, 125), id=57, color=[255, 51, 51]),
    58: dict(link=(125, 126), id=58, color=[255, 51, 51]),
    59: dict(link=(126, 127), id=59, color=[255, 51, 51]),
    60: dict(link=(127, 128), id=60, color=[255, 51, 51]),
    61: dict(link=(112, 129), id=61, color=[0, 255, 0]),
    62: dict(link=(129, 130), id=62, color=[0, 255, 0]),
    63: dict(link=(130, 131), id=63, color=[0, 255, 0]),
    64: dict(link=(131, 132), id=64, color=[0, 255, 0]),
}

coco_wholebody_right_hand_joint_indices = list(
    range(
        COCO_WHOLEBODY_KEYPOINTS.index("right_hand_root"),
        COCO_WHOLEBODY_KEYPOINTS.index("right_pinky") + 1,
    )
)
coco_wholebody_left_hand_joint_indices = list(
    range(
        COCO_WHOLEBODY_KEYPOINTS.index("left_hand_root"),
        COCO_WHOLEBODY_KEYPOINTS.index("left_pinky") + 1,
    )
)

#########################################
# Models Initialization
#########################################

enable_offload = not bool(os.getenv("DISABLE_OFFLOAD"))

with open("data/conversions/smpl_to_smplx.pkl", "rb") as f:
    mapping = torch.tensor(pickle.load(f)["matrix"]).float()

with open("data/conversions/smplx_to_smpl.pkl", "rb") as f:
    inv_mapping = torch.tensor(pickle.load(f)["matrix"]).float()

bm_in = BodyModel("smpl", "neutral")
bm_out = BodyModel("smplx", "neutral")
smpl2smplx = BodyConverter(bm_in, bm_out).cuda()
smpl2smplx = torch.jit.script(
    smpl2smplx
)  # optional: compile the converter for faster execution
smplx_layer = smplx.SMPLXLayer(
    model_path="data/body_models/smplx/SMPLX_NEUTRAL.npz",
    num_betas=10,
    use_face_contour=True,
).cuda()

normalize_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_chmr():
    global estimator
    if "estimator" not in globals() or estimator is None:
        estimator = HumanMeshEstimator()
        if enable_offload:
            estimator.model.cpu()
            estimator.cam_model.cpu()
        else:
            estimator.model.cuda()
            estimator.cam_model.cuda()
    return estimator


def delete_chmr():
    global estimator
    if "estimator" in globals() and estimator is not None:
        del estimator
        estimator = None
        torch.cuda.empty_cache()


def load_deco():
    global deco_model
    if "deco_model" not in globals() or deco_model is None:
        deco_model = DECO("hrnet", True, "cpu")
        checkpoint = torch.load("data/deco/deco_best.pth")
        deco_model.load_state_dict(checkpoint["deco"], strict=True)
        deco_model.eval()
        if not enable_offload:
            deco_model.cuda()
    return deco_model


def delete_deco():
    global deco_model
    if "deco_model" in globals() and deco_model is not None:
        del deco_model
        deco_model = None
        torch.cuda.empty_cache()


def load_wilor():
    global pipe_hand
    if "pipe_hand" not in globals() or pipe_hand is None:
        pipe_hand = WiLorHandPose3dEstimationPipeline(
            device="cuda", dtype=torch.float16, verbose=False
        )
        if enable_offload:
            pipe_hand.wilor_model.cpu()
            pipe_hand.hand_detector.cpu()
        else:
            pipe_hand.wilor_model.cuda()
            pipe_hand.hand_detector.cuda()
    return pipe_hand


def delete_wilor():
    global pipe_hand
    if "pipe_hand" in globals() and pipe_hand is not None:
        del pipe_hand
        pipe_hand = None
        torch.cuda.empty_cache()


torch.cuda.empty_cache()

smplx_indices = torch.tensor(
    [
        ORIGINAL_SMPLX_JOINT_NAMES.index(joint)
        for joint in COCO_WHOLEBODY_KEYPOINTS
        if joint in ORIGINAL_SMPLX_JOINT_NAMES
    ],
    dtype=torch.long,
)
coco_mask = torch.tensor(
    [joint in ORIGINAL_SMPLX_JOINT_NAMES for joint in COCO_WHOLEBODY_KEYPOINTS],
    dtype=torch.bool,
)
assert len(smplx_indices) == coco_mask.sum(), (
    f"Mismatch between SMPL-X indices and COCO mask: {len(smplx_indices)} vs {coco_mask.sum()}"
)


def convert_smplx_to_coco_wholebody(joints):
    """
    Converts joint coordinates from the SMPL-X format to the COCO WholeBody keypoint format.
    This function maps each keypoint in the COCO WholeBody format to the corresponding joint in the SMPL-X representation.
    For keypoints that have a corresponding joint name in the original SMPL-X joint list (ORIGINAL_SMPLX_JOINT_NAMES),
    their coordinates are copied into the new tensor. For keypoints without a corresponding match, a mismatch counter is incremented.
    An assertion is raised if the total number of mismatches does not equal 2, ensuring that the expected number of unmatched joints
    remains consistent.
    Parameters:
        joints (torch.Tensor): A tensor of shape (N, num_original_joints, 2), where N is the number of samples.
                               This tensor contains the joint coordinates (x, y) from the SMPL-X format. The tensor's device is preserved.
    Returns:
        torch.Tensor: A tensor of shape (N, len(COCO_WHOLEBODY_KEYPOINTS), 2) containing the joint coordinates in the COCO WholeBody format.
    Raises:
        AssertionError: If the number of mismatched joints is not exactly 2.
    """

    mismatch_joints = 0
    joints_coco = torch.zeros(
        joints.shape[0], len(COCO_WHOLEBODY_KEYPOINTS), 2, device=joints.device
    )
    # for i, joint_name in enumerate(COCO_WHOLEBODY_KEYPOINTS):
    #     if joint_name in ORIGINAL_SMPLX_JOINT_NAMES:
    #         joints_coco[:, i] = joints[:, ORIGINAL_SMPLX_JOINT_NAMES.index(joint_name)]
    #     else:
    #         mismatch_joints += 1

    # assert mismatch_joints == 2, f"Expected 2 mismatched joints, but got {mismatch_joints}."
    joints_coco[:, coco_mask] = joints[:, smplx_indices]
    return joints_coco


def get_point_visibility_mask(mask, points_2d):
    """
    Return a boolean tensor indicating the visibility of input 2D points based on a mask image.

    Parameters:
        mask (np.ndarray): The mask image as a numpy array of shape (H, W), with binary values (0 or 255).
        points_2d (torch.Tensor): A tensor of 2D points with shape (..., 2) representing (x, y) coordinates.

    Returns:
        torch.Tensor: A boolean tensor of the same shape as points_2d[..., 0], where True indicates
        the keypoint is within the image bounds and the corresponding mask value is non-zero.
    """

    height, width = mask.shape[:2]
    # Get x and y coordinates rounded to nearest integer.
    xs = torch.round(points_2d[..., 0]).long()
    ys = torch.round(points_2d[..., 1]).long()
    # Determine valid where coordinates are inside image bounds.
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    # Convert the numpy mask to a torch tensor.
    visibility_mask = torch.from_numpy(mask).to(points_2d.device)
    # Prepare a boolean tensor indicating if keypoints are inside the valid mask.
    in_mask = torch.zeros_like(valid, dtype=torch.bool)
    in_mask[valid] = visibility_mask[ys[valid], xs[valid]] != 0

    return in_mask


def compute_camera_facing_mask(vertices, faces, angle_threshold=70):
    """
    Compute a binary mask for each mesh in a batch.

    Parameters:
        vertices (np.ndarray): Array of vertices with shape (B, N, 3) or (N, 3).
        faces (np.ndarray): Array of face indices with shape (M, 3).
        angle_threshold (float): Maximum angle (in degrees) between a vertex normal and (0, 0, -1)
                                 for the vertex to be selected. Default is 70.

    Returns:
        np.ndarray: Binary mask of shape (B, N) where each vertex is True if the computed vertex
                    normal is at an angle less than or equal to angle_threshold with (0, 0, -1).
        np.ndarray: Computed vertex normals.
    """

    # Ensure vertices have a batch dimension.
    if vertices.ndim == 2:
        vertices = vertices[None, ...]
    B = vertices.shape[0]

    # Compute face normals for each face in each batch sample.
    v0 = vertices[:, faces[:, 0], :]  # (B, M, 3)
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    face_normals = np.cross(v1 - v0, v2 - v0)  # (B, M, 3)

    # Normalize face normals.
    face_norms = np.linalg.norm(face_normals, axis=2, keepdims=True) + 1e-10
    face_normals_normalized = face_normals / face_norms  # (B, M, 3)

    # Initialize an array for vertex normals with shape (B, N, 3)
    vertex_normals = np.zeros_like(vertices)
    for i in range(B):
        # Repeat each face normal along a new axis for accumulation.
        repeated_normals = np.tile(
            face_normals_normalized[i][:, np.newaxis, :], (1, 3, 1)
        )
        np.add.at(vertex_normals[i], faces, repeated_normals)

    # Normalize the accumulated vertex normals.
    v_norms = np.linalg.norm(vertex_normals, axis=2, keepdims=True) + 1e-10
    vertex_normals = vertex_normals / v_norms  # (B, N, 3)

    # Reference vector (0, 0, -1) and threshold.
    ref_vector = np.array([0, 0, -1])
    cos_threshold = np.cos(np.deg2rad(angle_threshold))

    # Compute dot products between vertex normals and the reference vector.
    dots = np.dot(vertex_normals, ref_vector)  # shape: (B, N)

    # Each vertex is selected if the angle (via cosine) is within the threshold.
    mask = dots >= cos_threshold

    return mask, vertex_normals


def get_body_model(body_model):
    """
    Retrieve a body model layer based on the specified model identifier.
    This function initializes and returns a body model layer depending on the provided
    body_model string. It supports two model types:
        - "SMPL": Returns an instance of the SMPLLayer using the SMPL_NEUTRAL.pkl file.
        - "SMPLX": Returns an instance of the SMPLXLayer using the SMPLX_NEUTRAL.npz file.
    Parameters:
            body_model (str): A string specifying the body model type. It must be either "SMPL" or "SMPLX".
    Returns:
            An instance of the corresponding body model layer:
                - For "SMPL": an object of type smplx.SMPLLayer.
                - For "SMPLX": an object of type smplx.SMPLXLayer.
    Note:
            The function assumes that the required model file exists at the predefined path:
                - "data/body_models/smpl/SMPL_NEUTRAL.pkl" for SMPL,
                - "data/body_models/smplx/SMPLX_NEUTRAL.npz" for SMPLX.
    """

    if body_model == "SMPL":
        return smplx.SMPLLayer(
            model_path="data/body_models/smpl/SMPL_NEUTRAL.pkl", num_betas=10
        )
    elif body_model == "SMPLX":
        return smplx_layer
    else:
        raise ValueError(
            "Invalid body model type. Supported types are 'SMPL' and 'SMPLX'."
        )


def get_body_model(body_model):
    """
    Retrieve a body model layer based on the specified model identifier.
    This function initializes and returns a body model layer depending on the provided
    body_model string. It supports two model types:
        - "SMPL": Returns an instance of the SMPLLayer using the SMPL_NEUTRAL.pkl file.
        - "SMPLX": Returns an instance of the SMPLXLayer using the SMPLX_NEUTRAL.npz file.
    Parameters:
            body_model (str): A string specifying the body model type. It must be either "SMPL" or "SMPLX".
    Returns:
            An instance of the corresponding body model layer:
                - For "SMPL": an object of type smplx.SMPLLayer.
                - For "SMPLX": an object of type smplx.SMPLXLayer.
    Note:
            The function assumes that the required model file exists at the predefined path:
                - "data/body_models/smpl/SMPL_NEUTRAL.pkl" for SMPL,
                - "data/body_models/smplx/SMPLX_NEUTRAL.npz" for SMPLX.
    """

    if body_model == "SMPL":
        return smplx.SMPLLayer(
            model_path="data/body_models/smpl/SMPL_NEUTRAL.pkl", num_betas=10
        )
    elif body_model == "SMPLX":
        return smplx_layer
    else:
        raise ValueError(
            "Invalid body model type. Supported types are 'SMPL' and 'SMPLX'."
        )


def get_ray_mesh_intersections_partwise(mesh: trimesh.Trimesh, part_segments: list):
    """
    Computes ray-mesh intersections for each specified part segment.

    Parameters:
        mesh (trimesh.Trimesh): The mesh on which rays will be cast.
        part_segments (list): A list of lists, where each sublist contains the vertex indices
                              corresponding to a specific part segment of the mesh.

    Returns:
        list: A list of dictionaries, one per part segment. Each dictionary contains:
            - 'locations': Intersection points on the mesh for the rays belonging to the segment.
            - 'dists': Distances from the ray origins to the intersection points.
            - 'index_ray': Indices of the rays within the segment (relative to the segment).
            - 'index_tri': Indices of the intersected triangles.
    """

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    # Create a ray for each part vertex
    all_part_verts = mesh.vertices[np.concatenate(part_segments)]
    ray_origins = np.zeros((all_part_verts.shape[0], 3))
    ray_directions = all_part_verts / np.linalg.norm(all_part_verts, axis=1)[:, None]

    # Find the intersection of the rays with the mesh
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
    )

    # Compute distances between the intersection locations and their corresponding part vertices
    dists = np.linalg.norm(locations - all_part_verts[index_ray], axis=1)

    # Split the intersection results (locations, dists, index_ray, index_tri) per part segment.
    segment_results = []
    offset = 0
    for seg in part_segments:
        seg_len = len(seg)
        # Identify which rays from the concatenated array belong to the current segment.
        mask = (index_ray >= offset) & (index_ray < offset + seg_len)
        segment_results.append(
            {
                "locations": locations[mask],
                "dists": dists[mask],
                "index_ray": index_ray[mask] - offset,
                "index_tri": index_tri[mask],
            }
        )
        offset += seg_len

    return segment_results


def get_self_occlusion_mask(vertices: np.ndarray, faces: np.ndarray, threshold: float):
    """
    Computes a self-occlusion mask for limb segments by shooting rays from the vertices
    of each segment and measuring their visibility.

    For each segment in the mesh (defined by the global variable part_segments), the
    function casts rays in the direction of each vertex. It then computes the fraction
    of vertices that are considered visible based on either a direct hit (distance < 1e-6)
    or an intersection with a face that belongs to the same segment. Segments with a
    visible fraction lower than the provided threshold are flagged as self-occluded.

    Parameters:
        vertices (np.ndarray): Array of mesh vertices.
        faces (np.ndarray): Array of mesh faces.
        threshold (float): The fraction of visible vertices required to not be considered occluded.

    Returns:
        visibility (np.ndarray): Boolean mask of vertices that are self-occluded.
        fracs_visible (list): List of visible fractions for each segment.
    """

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    segment_results = get_ray_mesh_intersections_partwise(mesh, part_segments)

    visibility = np.zeros(vertices.shape[0], dtype=bool)

    fracs_visible = []
    for seg_idx, seg_res in enumerate(segment_results):
        visible_verts = 0

        intersected_idxs, dists, intersected_tri = (
            seg_res["index_ray"],
            seg_res["dists"],
            seg_res["index_tri"],
        )
        for idx, dist, tri in zip(intersected_idxs, dists, intersected_tri):
            if dist < 1e-6:
                visible_verts += 1
            elif set(mesh.faces[tri]).intersection(set(part_segments[seg_idx])):
                visible_verts += 1

        frac = visible_verts / len(part_segments[seg_idx])
        fracs_visible.append(frac)
        if frac < threshold:
            visibility[part_segments[seg_idx]] = True

    return visibility, fracs_visible


def process_single_mesh(args):
    """
    Helper function to compute the self-occlusion mask for a single mesh below.
    """
    v, faces, threshold = args
    # v has shape (N, 3); get the occlusion mask for one mesh
    mask, _ = get_self_occlusion_mask(v, faces, threshold)
    return mask


def get_self_occlusion_masks(
    vertices: torch.Tensor,
    faces: np.ndarray,
    threshold: float = 0.7,
    parallel: bool = False,
):
    """
    Compute self-occlusion masks for a batch of meshes.

    Parameters:
        vertices (torch.Tensor): Batch of vertices with shape (B, N, 3).
        faces (np.ndarray): Array of face indices defining the mesh.
        threshold (float): Visibility threshold to determine occlusion.
        parallel (bool): Flag to enable parallel processing.

    Returns:
        np.ndarray: Batch of occlusion masks with shape (B, N).

    Note:
        The function is faster when parallel=False for small batch sizes (B < 30).
    """

    # batch of vertices, (B, N, 3)
    vertices = vertices.detach().cpu().numpy()

    B = vertices.shape[0]
    args = [(vertices[i], faces, threshold) for i in range(B)]
    if parallel:
        with Pool() as pool:
            results = pool.map(process_single_mesh, args)
    else:
        results = [process_single_mesh(arg) for arg in args]

    mask_batch = np.stack(results, axis=0)  # (B, N) mask
    return mask_batch

def get_hand_pose(
    image: Image, keypoints_image: np.array, keypoint_scores_image: np.array
):
    """
    Estimate the 3D hand pose(s) for each detected human in the image.
    Parameters:
        image (PIL.Image): The input image.
        keypoints_image (numpy.ndarray): Array of keypoints for each human detected in the image.
        keypoint_scores_image (numpy.ndarray): Array of keypoint confidence scores corresponding to keypoints_image.
    Returns:
        List[dict]: A list where each element is a dictionary for a detected human containing:
            - 'left_hand': Estimated hand pose information for the left hand (or None if not detected).
            - 'right_hand': Estimated hand pose information for the right hand (or None if not detected).
    Note:
        The function uses a hand pose estimation pipeline (pipe_hand) to detect and predict hand poses.
    """

    if enable_offload:
        pipe_hand.wilor_model.cuda()
        pipe_hand.hand_detector.cuda()

    image_np = np.array(image)
    hand_bbox_thresh = 0.7
    n_humans = len(keypoints_image)

    is_rights = []
    hand_bboxes = []
    human_idx = []

    # reference: https://github.com/hongsukchoi/HSfM_RELEASE/blob/5221f002f42608afbbc5b67724f943b9f9f2b2b1/get_mano_wilor_for_hsfm.py#L444
    for human in range(n_humans):
        left_hand_joints = keypoints_image[human][
            coco_wholebody_left_hand_joint_indices
        ]
        right_hand_joints = keypoints_image[human][
            coco_wholebody_right_hand_joint_indices
        ]

        left_hand_scores = keypoint_scores_image[human][
            coco_wholebody_left_hand_joint_indices
        ]
        right_hand_scores = keypoint_scores_image[human][
            coco_wholebody_right_hand_joint_indices
        ]

        if left_hand_scores.mean() > hand_bbox_thresh:
            left_hand_bbox = np.array(
                [*left_hand_joints.min(axis=0), *left_hand_joints.max(axis=0)]
            )

            hand_bboxes.append(left_hand_bbox)
            is_rights.append(0)
            human_idx.append(human)

        if right_hand_scores.mean() > hand_bbox_thresh:
            right_hand_bbox = np.array(
                [*right_hand_joints.min(axis=0), *right_hand_joints.max(axis=0)]
            )

            hand_bboxes.append(right_hand_bbox)
            is_rights.append(1)
            human_idx.append(human)

    len(is_rights), len(hand_bboxes), len(human_idx)

    outputs = pipe_hand.predict_with_bboxes(
        np.array(image),
        bboxes=np.array(hand_bboxes, dtype=np.float32),
        is_rights=is_rights,
    )

    results = [{"left_hand": None, "right_hand": None} for _ in range(n_humans)]

    for hand_idx, out in enumerate(outputs):

        hand_bbox = np.array(out["hand_bbox"], dtype=np.float32)  # (4,), x1y1x2y2 list
        saving_output = {
            "hand_bbox": hand_bbox,  # (4,)
            "global_orient": out["wilor_preds"]["global_orient"][0],  # (1, 3)
            "hand_pose": out["wilor_preds"]["hand_pose"][0],  # (15, 3)
            "betas": out["wilor_preds"]["betas"][0],  # (10,)
            "pred_keypoints_3d": out["wilor_preds"]["pred_keypoints_3d"][0],  # (21, 3)
            "pred_keypoints_2d": out["wilor_preds"]["pred_keypoints_2d"][0],  # (21, 3)
        }
        person_id = human_idx[hand_idx]

        is_right_from_vitpose = is_rights[hand_idx]
        is_right_from_wilor = out["is_right"]
        if is_right_from_vitpose != is_right_from_wilor:
            print(f"Mismatch between vitpose and wilor for person {person_id}")
            continue

        if is_right_from_vitpose:
            results[person_id]["right_hand"] = saving_output
        else:
            results[person_id]["left_hand"] = saving_output

    if enable_offload:
        pipe_hand.wilor_model.cpu()
        pipe_hand.hand_detector.cpu()
    torch.cuda.empty_cache()

    return results


def process_human(
    image_np: np.array,
    body_model: str,
    boxes: np.array = None,
    confs: np.array = None,
    keypoints: np.array = None,
    keypoint_scores: np.array = None,
    smpl_model="chmr",
):
    """
    Process an image to estimate human body parameters and mesh, supporting SMPLX model with hand pose extraction.
        Args:
            image_np (np.array): The input image as a NumPy array, in RGB format.
            body_model (str): The type of body model to use. Currently, "SMPLX" and "SMPL" are supported.
            boxes (np.array, optional): Detection boxes (N, 4) for human figures. Defaults to None.
            confs (np.array, optional): Confidence scores (N,) corresponding to the detection boxes. Defaults to None.
            keypoints (np.array, optional): Human keypoints (N, N_joints, 2) used for augmenting SMPLX estimation. Required when body_model is "SMPLX".
            keypoint_scores (np.array, optional): Scores for the human keypoints (N, N_joints) . Required when body_model is "SMPLX".
            smpl_model (str, optional): The SMPL model to use. Currently, "chmr" is supported. Defaults to "chmr".
        Returns:
            tuple: A tuple containing:
                - mesh (trimesh.Trimesh): The reconstructed 3D human mesh of the most confident person detection.
                - cam_intr: Camera intrinsics.
                - out_body_params (dict): Dictionary with body pose parameters including body_pose, global_orient, betas, and hand poses.
                - output_cam_trans: Camera translation parameters.
                - boxes (np.array): Returned original boxes, or detected boxes if not provided.
                - confs (np.array): Returned original confidences, or detected confidences if not provided.
    """
    
    if smpl_model == "chmr":
        mesh, cam_intr, out_body_params, output_cam_trans, boxes, confs = (
            get_chmr_results(image_np, boxes, confs)
        )
    else:
        raise ValueError(f"Unsupported SMPL model: {smpl_model}")

    if body_model == "SMPLX":
        assert keypoints is not None and keypoint_scores is not None, (
            "Keypoints and keypoint scores are required for SMPLX model."
        )

        n_humans = len(boxes)
        thetas = rotmat_to_aa(out_body_params["body_pose"].reshape(-1, 3, 3)).reshape(
            n_humans, -1
        )
        betas = out_body_params["betas"]
        global_orients = rotmat_to_aa(
            out_body_params["global_orient"].reshape(-1, 3, 3)
        ).reshape(n_humans, -1)

        pose = torch.cat([global_orients, thetas], dim=1)
        betas = betas
        cam_trans = output_cam_trans

        out = smpl2smplx.convert(pose, betas, cam_trans)

        # Extract the MANO hand pose for each person's left and right hands
        results = get_hand_pose(Image.fromarray(image_np), keypoints, keypoint_scores)

        out_posevecs = out["pose_rotvecs"]
        out_global_orient = aa_to_rotmat(out_posevecs[:, :3])
        out_body_pose = aa_to_rotmat(
            out_posevecs[:, (1 * 3) : (22 * 3)].reshape(-1, 3)
        ).reshape(n_humans, 21, 3, 3)
        out_lhand_pose = aa_to_rotmat(
            out_posevecs[:, (25 * 3) : (40 * 3)].reshape(-1, 3)
        ).reshape(n_humans, 15, 3, 3)
        out_rhand_pose = aa_to_rotmat(
            out_posevecs[:, (40 * 3) : (55 * 3)].reshape(-1, 3)
        ).reshape(n_humans, 15, 3, 3)

        # if any of the hand results are available, update the hand pose
        for person_id, result in enumerate(results):
            if result["left_hand"] is not None:
                out_lhand_pose[person_id] = aa_to_rotmat(
                    torch.tensor(result["left_hand"]["hand_pose"]).to(
                        out_lhand_pose.device
                    )
                ).reshape(15, 3, 3)
            if result["right_hand"] is not None:
                out_rhand_pose[person_id] = aa_to_rotmat(
                    torch.tensor(result["right_hand"]["hand_pose"]).to(
                        out_rhand_pose.device
                    )
                ).reshape(15, 3, 3)

        most_confident_idx = np.argmax(confs)

        smplx_output = smplx_layer(
            betas=(out["shape_betas"][most_confident_idx].unsqueeze(0)),
            global_orient=out_global_orient[most_confident_idx].unsqueeze(0),
            body_pose=out_body_pose[most_confident_idx].unsqueeze(0),
            left_hand_pose=out_lhand_pose[most_confident_idx].unsqueeze(0),
            right_hand_pose=out_rhand_pose[most_confident_idx].unsqueeze(0),
        )

        vertices = (
            (smplx_output.vertices[0] + out["trans"][most_confident_idx])
            .detach()
            .cpu()
            .numpy()
        )
        mesh = trimesh.Trimesh(vertices, smplx_layer.faces, process=False)

        out_body_params = {
            "body_pose": out_body_pose,
            "global_orient": out_global_orient,
            "betas": out["shape_betas"],
            "left_hand_pose": out_lhand_pose,
            "right_hand_pose": out_rhand_pose,
        }

        output_cam_trans = out["trans"]

    return mesh, cam_intr, out_body_params, output_cam_trans, boxes, confs


def get_chmr_results(
    image_np: np.array, boxes: np.array = None, confs: np.array = None
):
    """ """
    if enable_offload:
        estimator.model.cuda()
        estimator.cam_model.cuda()

    estimator.model.eval()

    img_cv2 = image_np[:, :, ::-1]

    if boxes is None:
        assert not confs, (
            "If boxes are not provided, confs should not be provided either."
        )
        # Detect humans in the image
        det_out = estimator.detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        confs = det_instances.scores[valid_idx].cpu().numpy()
    else:
        assert confs is not None, (
            "If boxes are provided, confs should also be provided."
        )

    bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
    bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

    # Get Camera intrinsics using HumanFoV Model
    cam_intr = estimator.get_cam_intrinsics(img_cv2)
    dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_intr, False, img_path=None)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=10
    )

    batch = next(iter(dataloader))
    batch = recursive_to(batch, "cuda")
    img_h, img_w = batch["img_size"][0]
    with torch.no_grad():
        out_smpl_params, out_cam, focal_length_ = estimator.model(batch)

    output_vertices, output_joints, output_cam_trans = estimator.get_output_mesh(
        out_smpl_params, out_cam, batch
    )

    focal_length = (focal_length_[0], focal_length_[0])
    pred_vertices_array = (
        (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
    )

    most_confident_idx = np.argmax(confs)

    mesh = trimesh.Trimesh(
        pred_vertices_array[most_confident_idx],
        estimator.smpl_model.faces,
        process=False,
    )

    if enable_offload:
        estimator.model.cpu()
        estimator.cam_model.cpu()

    torch.cuda.empty_cache()

    return mesh, cam_intr, out_smpl_params, output_cam_trans, boxes, confs

def resize_and_pad(image_crop, target_size=256):
    h, w = image_crop.shape[:2]
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    return padded

def convert_contacts(contact_labels, mapping):
    """
    Converts the contact labels from SMPL to SMPL-X format and vice-versa.
    Taken from: https://github.com/sha2nkt/deco/blob/main/reformat_contacts.py

    Args:
        contact_labels: contact labels in SMPL or SMPL-X format
        mapping: mapping from SMPL to SMPL-X vertices or vice-versa

    Returns:
        contact_labels_converted: converted contact labels
    """
    bs = contact_labels.shape[0]
    mapping = mapping[None].expand(bs, -1, -1)
    contact_labels_converted = torch.bmm(mapping, contact_labels[..., None])
    contact_labels_converted = contact_labels_converted.squeeze()
    return contact_labels_converted

def convert_contacts(contact_labels, mapping):
    """
    Converts the contact labels from SMPL to SMPL-X format and vice-versa.
    Taken from: https://github.com/sha2nkt/deco/blob/main/reformat_contacts.py

    Args:
        contact_labels: contact labels in SMPL or SMPL-X format
        mapping: mapping from SMPL to SMPL-X vertices or vice-versa

    Returns:
        contact_labels_converted: converted contact labels
    """
    bs = contact_labels.shape[0]
    mapping = mapping[None].expand(bs, -1, -1)
    contact_labels_converted = torch.bmm(mapping, contact_labels[..., None])
    contact_labels_converted = contact_labels_converted.squeeze(dim=-1)
    return contact_labels_converted

def get_contact_probs(
    images_cv: List[np.array],
    boxes: List[np.array],
    body_model,
    expand_box=0.1,
    max_batch_size=32,
):
    """
    Extracts contact probability predictions for image regions defined by bounding boxes.
    This function processes the input image by extracting regions specified by the provided
    bounding boxes. Each region is resized, normalized, and then forwarded through a pretrained
    model (deco_model) to obtain contact probabilities. The model computations are performed on
    the GPU for efficiency, and the final output is returned as a NumPy array after detaching
    from the PyTorch computation graph.
    Parameters:
        image_cv (np.array): The source image in a NumPy array format (typically in BGR order as read by OpenCV).
        boxes (np.array): An array of bounding boxes where each box is represented by
                          [x1, y1, x2, y2]. Each box defines the coordinates for a sub-region of the image.
        body_model (str): A string specifying the body model type. It must be either "SMPL" or "SMPLX".
        expand_box (float): A float value specifying the amount by which to expand the bounding box
                            region before extracting the image crop.
    Returns:
        np.array: A NumPy array containing the contact probabilities predicted by the model for each
                  image crop.
    """

    assert len(images_cv) == len(boxes), "Number of images and boxes must match."
    humans_per_image = [i.shape[0] for i in boxes]

    model = deco_model
    if enable_offload:
        model.cuda()

    image_crops = []
    for image_cv, boxes in zip(images_cv, boxes):
        crops = [
            image_cv[
                max(int(y1 - expand_box * (y2 - y1)), 0) : min(
                    int(y2 + expand_box * (y2 - y1)), image_cv.shape[0]
                ),
                max(int(x1 - expand_box * (x2 - x1)), 0) : min(
                    int(x2 + expand_box * (x2 - x1)), image_cv.shape[1]
                ),
            ]
            for x1, y1, x2, y2 in boxes.tolist()
        ]
        image_crops.extend(crops)

    image_crops = [resize_and_pad(image_crop) for image_crop in image_crops]
    image_crops = [image_crop.transpose(2, 0, 1) / 255.0 for image_crop in image_crops]
    image_crops = [
        normalize_img(torch.from_numpy(image_crop).float())
        for image_crop in image_crops
    ]
    image_crops = torch.stack(image_crops).cuda()

    outputs = []
    for i in range(0, image_crops.size(0), max_batch_size):
        batch = image_crops[i : i + max_batch_size]
        cont_batch, _, _ = model(batch)
        outputs.append(cont_batch)

    cont = torch.cat(outputs, dim=0)
    cont = cont.detach()  # .cpu().numpy()

    if body_model == "SMPLX":
        cont = convert_contacts(cont, mapping.to(cont.device))

    # reshape the contact probabilities to match the number of humans in each image
    split_cont = []
    offset = 0
    for n in humans_per_image:
        split_cont.append(cont[offset : offset + n])
        offset += n

    # contact_vertices = []
    # for i in range(len(boxes)):
    #     conts = cont[i] >= 0.5
    #     indices = np.nonzero(conts)
    #     contact_vertices.append(indices)
    # cont = cont >= 0.5

    if enable_offload:
        model.cpu()
    torch.cuda.empty_cache()

    return split_cont

def get_static_contacts(body_model, n_humans):
    """
    Computes static contact labels for a given body model and number of human instances.

    This function loads predefined contact vertices for various body parts from JSON files
    located in the "data/body_segments" directory. For each specified body part (e.g., 'L_Leg',
    'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs'), it reads the vertex indices and
    collects them into a unified list of unique contact vertices. It then creates a contact label
    tensor of shape (n_humans, 10475) (with the assumption that the model has 10475 vertices),
    initializing it with zeros on the CUDA device, and sets the contact indices to 1. If the
    provided body model is "SMPL", the contact tensor is converted to the SMPL format using an
    inverse mapping.

    The contact labels are provided by PROX (https://prox.is.tue.mpg.de/).

    Parameters:
        body_model (str): The type of body model, e.g., "SMPL". Determines if conversion of
                          the contact labels is required.
        n_humans (int): The number of human instances for which to generate the contact label tensor.

    Returns:
        torch.Tensor: A tensor of shape (n_humans, 10475) with contact labels set to 1 for the
                      defined contact vertices. If the body model is "SMPL", the tensor is
                      converted to the SMPL vertex ordering.
    """

    all_contact_vertices = []
    for part in ["L_Leg", "R_Leg", "L_Hand", "R_Hand", "gluteus", "back", "thighs"]:
        with open(f"data/body_segments/{part}.json", "r") as f:
            data = json.load(f)
            all_contact_vertices.extend(list(set(data["verts_ind"])))

    print("Total number of contact vertices:", len(all_contact_vertices))

    # The contact labels are defined on the SMPL-X model with 10475 vertices
    contacts = torch.zeros(n_humans, 10475).to("cuda")
    contacts[:, all_contact_vertices] = 1

    # Convert the contact labels to SMPL format if the body model is SMPL
    if body_model == "SMPL":
        contacts = convert_contacts(contacts, inv_mapping.to(contacts.device))

    return contacts

def get_contact_mask(
    scene_mesh, human_vertices, threshold=0.02, interpenetration_threshold=0.15
):
    """
    Given a scene mesh and human vertices, compute a boolean mask indicating
    whether each vertex is within the threshold distance from the mesh surface.
    This approach accounts for large faces by measuring distance to the surface,
    not just the nearest vertices.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene_o3d = o3d.geometry.TriangleMesh()
    scene_o3d.vertices = o3d.utility.Vector3dVector(scene_mesh.vertices)
    scene_o3d.triangles = o3d.utility.Vector3iVector(scene_mesh.faces)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene_o3d))
    ret_dict = scene.compute_closest_points(
        o3d.core.Tensor.from_numpy(human_vertices.astype(np.float32))
    )
    points = ret_dict["points"].numpy()
    distances = np.linalg.norm(points - human_vertices, axis=1)

    normals = ret_dict["primitive_normals"].numpy()  # shape (N, 3)
    directions = (
        human_vertices - points
    )  # vector from the closest point to the human vertex
    directions_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    # avoid division by zero
    directions_normalized = directions / (directions_norm + 1e-6)
    dot_products = np.sum(directions_normalized * normals, axis=1)
    contact_from_dot = dot_products < 0
    # mark as contact if within threshold or if the normal check indicates contact
    mask = (distances <= threshold) | (
        contact_from_dot & (distances < interpenetration_threshold)
    )

    return mask

def get_contact_mask_from_pointcloud(
    scene_points,
    scene_normals,
    human_vertices,
    threshold=0.02,
    interpenetration_threshold=0.15,
):
    """
    Given a scene point cloud with corresponding normals and human vertices,
    compute a boolean mask indicating contact vertices. The function finds,
    for each human vertex, the nearest scene point, and marks it as in contact
    if the distance is within the threshold or if the direction to the human
    vertex and the scene normal have a negative dot product.

    Parameters:
        scene_points (np.ndarray): An array of shape (N, 3) of scene point coordinates.
        scene_normals (np.ndarray): An array of shape (N, 3) of normals corresponding to scene_points.
        human_vertices (np.ndarray): An array of shape (M, 3) of human vertex coordinates.
        threshold (float): Distance threshold to consider a vertex in contact.

    Returns:
        np.ndarray: A boolean array of length M, where True indicates a contact vertex.
    """

    tree = KDTree(scene_points)
    distances, indices = tree.query(human_vertices)
    distances = distances.squeeze()
    indices = indices.squeeze()
    nearest_pts = scene_points[indices]
    nearest_normals = scene_normals[indices]

    # Compute the direction from the nearest scene point to the human vertex
    directions = human_vertices - nearest_pts
    norm_directions = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6
    direction_normalized = directions / norm_directions

    dot_products = np.sum(direction_normalized * nearest_normals, axis=1)
    contact_from_dot = (dot_products < 0) & (distances < interpenetration_threshold)

    mask = (distances <= threshold) | contact_from_dot
    return mask


def get_contact_mask_from_pointcloud_pt(
    scene_points,
    scene_normals,
    human_vertices,
    threshold=0.02,
    interpenetration_threshold=0.15,
    fg_mask=None,
):
    # Convert numpy arrays to torch tensors on GPU without a batch dimension.
    device = torch.device("cuda")
    scene_points_t = torch.tensor(
        scene_points, dtype=torch.float32, device=device
    )  # (N, 3)
    scene_normals_t = torch.tensor(
        scene_normals, dtype=torch.float32, device=device
    )  # (N, 3)
    human_vertices_t = torch.tensor(
        human_vertices, dtype=torch.float32, device=device
    )  # (M, 3)
    fg_mask_binary = (
        torch.tensor(fg_mask > 0, dtype=torch.bool, device=device)
        if fg_mask is not None
        else None
    )

    # Create Pointclouds objects.
    scene_points_pc = Pointclouds(scene_points_t.unsqueeze(0))
    query_pts_pc = Pointclouds(human_vertices_t.unsqueeze(0))

    # Perform knn search using pytorch3d's knn_points.
    nn_dists, nn_idxs, _ = knn_points(
        oputil.convert_pointclouds_to_tensor(query_pts_pc)[0],
        oputil.convert_pointclouds_to_tensor(scene_points_pc)[0],
        K=1,
        return_nn=False,
    )
    # Compute distances and indices.
    dists = nn_dists.sqrt().squeeze(0).squeeze(-1)  # shape: (M,)
    indices = nn_idxs.squeeze(0).squeeze(-1)  # shape: (M,)

    # Gather nearest scene points and corresponding normals.
    nearest_pts = scene_points_t[indices]  # (M, 3)
    nearest_normals = scene_normals_t[indices]  # (M, 3)

    # Compute directions from the nearest scene points to human vertices.
    directions = human_vertices_t - nearest_pts  # (M, 3)
    norms = torch.norm(directions, dim=1, keepdim=True) + 1e-6
    direction_normalized = directions / norms
    dot_products = (direction_normalized * nearest_normals).sum(dim=1)
    contact_from_dot = (dot_products < 0) & (dists < interpenetration_threshold)

    mask = (dists <= threshold) | contact_from_dot
    mask = mask.cpu().numpy()

    if fg_mask_binary is not None:
        assert scene_points_t.shape[0] == fg_mask_binary.shape[0], (
            "Scene points and foreground mask must have the same number of points."
        )

        nn_in_fg_mask = fg_mask_binary[
            indices
        ]  # is the nearest point of this vertex in the foreground mask?
        contact_nn = nn_in_fg_mask[
            mask
        ]  # get a mask of FG mask for each contact vertex
        return (
            mask,
            torch.any(contact_nn).cpu().numpy(),
        )  # return true if any of the contact points are in the foreground mask

    return mask
