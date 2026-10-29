"""On-disk format of the released per-frame predictions.

One ``.npz`` per evaluated frame. It holds *only* what the metrics need:

===========================  =========================================================
``init_*`` / ``final_*``     SMPL-X body parameters (betas + 6D rotations) and camera
                             translation, before and after scene-aware optimisation.
``K``                        3x3 pinhole intrinsics of the predicted camera.
``contact_deco``             (10475,) bool -- the image-based contact prediction
                             (DECO) that the optimiser was given, kept as a baseline.
``scene_points``             (M, 3) float32 -- reconstructed scene points, metric
                             scale, camera frame, floor-plane samples included.
``scene_normals``            (M, 3) float32 -- unit normal per scene point.
``visibility_mask``          (10475,) bool -- body vertices that project into the
                             segmented person in the input image (PROX only).
``contact_pred_reference``   (10475,) bool -- contact mask recomputed by the original
                             pipeline on the *full* scene cloud; the evaluation
                             re-derives it from ``scene_points`` and checks it matches.
===========================  =========================================================

No image, depth map, mesh or ground-truth field is stored: the released files
carry our predictions only. Ground truth is read from your own copy of PROX /
RICH at evaluation time.

``scene_points``/``scene_normals`` are a subset of the full reconstructed cloud
(see ``prepare/``). The subset always contains the nearest neighbour of every
body vertex, which is what the contact rule depends on, so the contact metrics
are bit-identical to running on the full cloud -- ``contact_pred_reference``
exists to verify exactly that.
"""

import numpy as np

BODY_PARAM_KEYS = ('betas', 'body_pose', 'global_orient', 'left_hand_pose', 'right_hand_pose')


def save_frame(path, init_params, init_cam_trans, final_params, final_cam_trans, K,
               contact_deco, scene_points, scene_normals, contact_pred_reference,
               visibility_mask=None):
    """Write one frame's stripped prediction file (see module docstring)."""
    payload = {
        'K': np.asarray(K, dtype=np.float32),
        'contact_deco': np.asarray(contact_deco).reshape(-1).astype(bool),
        'scene_points': np.asarray(scene_points, dtype=np.float32),
        'scene_normals': np.asarray(scene_normals, dtype=np.float32),
        'contact_pred_reference': np.asarray(contact_pred_reference).reshape(-1).astype(bool),
    }
    for key in BODY_PARAM_KEYS:
        payload[f'init_{key}'] = np.asarray(init_params[key], dtype=np.float32)
        payload[f'final_{key}'] = np.asarray(final_params[key], dtype=np.float32)
    payload['init_cam_trans'] = np.asarray(init_cam_trans, dtype=np.float32)
    payload['final_cam_trans'] = np.asarray(final_cam_trans, dtype=np.float32)
    if visibility_mask is not None:
        payload['visibility_mask'] = np.asarray(visibility_mask).reshape(-1).astype(bool)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_frame(path):
    """Read one frame's stripped prediction file into a plain dict."""
    with np.load(path) as data:
        frame = {
            'K': data['K'],
            'contact_deco': data['contact_deco'],
            'scene_points': data['scene_points'],
            'scene_normals': data['scene_normals'],
            'contact_pred_reference': data['contact_pred_reference'],
            'init_params': {k: data[f'init_{k}'] for k in BODY_PARAM_KEYS},
            'final_params': {k: data[f'final_{k}'] for k in BODY_PARAM_KEYS},
            'init_cam_trans': data['init_cam_trans'],
            'final_cam_trans': data['final_cam_trans'],
        }
        if 'visibility_mask' in data:
            frame['visibility_mask'] = data['visibility_mask']
    return frame
