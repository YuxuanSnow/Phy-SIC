"""SMPL-X helpers shared by the PROX and RICH evaluations."""

import numpy as np
import smplx
import torch

from .geometry import rot6d_to_rotmat


def load_smplx_layer(model_path, device='cuda', num_betas=10):
    """The rotation-matrix SMPL-X layer our predictions are expressed in."""
    return smplx.SMPLXLayer(model_path=str(model_path), num_betas=num_betas).to(device)


def pose_prediction(smplx_layer, body_params, cam_trans, device='cuda'):
    """Run the SMPL-X layer on one predicted frame.

    Args:
        body_params: dict with ``betas`` and 6D rotations, each with a leading
            batch dimension of 1.
        cam_trans: (1, 3) camera-frame translation.

    Returns:
        (vertices, joints) as ``(N, 3)`` / ``(J, 3)`` numpy arrays in the camera frame.
    """
    kwargs = {}
    for key, value in body_params.items():
        tensor = torch.as_tensor(np.asarray(value)).float().to(device)
        kwargs[key] = tensor if key == 'betas' else rot6d_to_rotmat(tensor)

    output = smplx_layer(**kwargs, transl=torch.as_tensor(np.asarray(cam_trans)).float().to(device))
    return (output.vertices[0].detach().cpu().numpy(),
            output.joints[0].detach().cpu().numpy())
