"""Rotation / projection helpers used by the evaluation scripts.

Copied verbatim (behaviour-wise) from the PhySIC code base so that this folder is
self-contained and can be released on its own.
"""

import numpy as np
import torch
import torch.nn.functional as F


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (B, 4) in (w, x, y, z) order -> rotation matrices (B, 3, 3)."""
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2],
                       dim=1).view(B, 3, 3)


def aa_to_rotmat(theta: torch.Tensor) -> torch.Tensor:
    """Axis-angle (B, 3) -> rotation matrices (B, 3, 3), via quaternions."""
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def rot6d_to_rotmat(rots: torch.Tensor) -> torch.Tensor:
    """6D rotation representation (Zhou et al., CVPR 2019) -> rotation matrices.

    Accepts (B, 6) or (B, N, 6) and returns (B, 3, 3) or (B, N, 3, 3).
    """
    x = rots.view(-1, 6) if rots.dim() == 3 else rots

    x = x.reshape(-1, 2, 3).contiguous()
    a1 = x[:, 0, :]
    a2 = x[:, 1, :]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rotmats = torch.stack((b1, b2, b3), dim=-1)

    if rots.dim() == 3:
        rotmats = rotmats.view(rots.shape[0], rots.shape[1], 3, 3)

    return rotmats


def apply_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    homog_points = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog_points @ transform.T)[:, :3]


def project_points_torch(points: torch.Tensor, cam_intr: torch.Tensor) -> torch.Tensor:
    """Project (N, 3) camera-space points with a 3x3 intrinsics matrix -> (N, 2)."""
    points_h = points / points[:, 2:3]
    projected_h = cam_intr @ points_h.T
    return projected_h[:2].T
