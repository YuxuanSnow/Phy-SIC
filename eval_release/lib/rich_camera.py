"""RICH camera calibration.

RICH ships one XML per camera holding a 3x4 extrinsics matrix (world -> camera)
and the intrinsics. The evaluation only needs the extrinsics, to bring the
ground-truth body/scene scans into the camera frame our predictions live in.

Adapted from the RICH toolkit's ``CalibratedCamera``.
"""

import xml.etree.ElementTree as ET

import numpy as np


def extract_cam_param_xml(xml_path):
    """Parse a RICH calibration XML.

    Returns:
        dict with ``focal_length_x``, ``focal_length_y``, ``center`` (2,),
        ``rotation`` (3, 3) and ``translation`` (3,).
    """
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]

    rotation = np.array([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]],
                         [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]],
                         [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]],
                        dtype=np.float32)
    translation = np.array([extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]],
                           dtype=np.float32)

    return {
        'focal_length_x': intrinsics_mat[0],
        'focal_length_y': intrinsics_mat[4],
        'center': np.array([intrinsics_mat[2], intrinsics_mat[5]], dtype=np.float32),
        'rotation': rotation,
        'translation': translation,
    }


def world_to_camera_transform(xml_path) -> np.ndarray:
    """4x4 world-to-camera transform for a RICH camera calibration file."""
    params = extract_cam_param_xml(xml_path)
    transform = np.eye(4)
    transform[:3, :3] = params['rotation']
    transform[:3, 3] = params['translation']
    return transform
