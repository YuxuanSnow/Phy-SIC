###
# 3D Conversions taken from CameraHMR (https://github.com/pixelite1201/CameraHMR/)
###

import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple

from pytorch3d.ops import utils as oputil
from pytorch3d.ops import knn_points
from pytorch3d.structures.pointclouds import Pointclouds


class kNNField(torch.nn.Module):
    # TODO: make use of the fact that the grid is a regular grid, and speed up the kNN grid computation initially.
    def __init__(
        self,
        points: torch.Tensor,
        resolution: int = 128,
        bounds: Tuple[torch.Tensor, torch.Tensor] = None,
        margin: float = 0.1,
    ):
        """
        Initialize a kNNField instance.

        Args:
            points (torch.Tensor): Tensor containing the input points.
            resolution (int, optional): Field resolution for quantization, default is 128.
            bounds (Tuple[torch.Tensor, torch.Tensor], optional): Tuple with lower and upper bounds of the data, default is None.
            margin (float, optional): Additional margin around the points, default is 0.1.
        """

        super(kNNField, self).__init__()
        self.points = points
        self.resolution = resolution
        self.margin = margin
        self.bounds = bounds
        self.precompute_knn()

    def precompute_knn(self):
        if self.bounds is not None:
            self.lower_bound, self.upper_bound = self.bounds
        else:
            lb = self.points.min(axis=0).values
            ub = self.points.max(axis=0).values

            self.lower_bound = lb - self.margin * (ub - lb)
            self.upper_bound = ub + self.margin * (ub - lb)

        print("Lower Bound: ", self.lower_bound)
        print("Upper Bound: ", self.upper_bound)
        print("Number of Points: ", self.points.shape[0])

        grid = torch.meshgrid(
            torch.linspace(self.lower_bound[0], self.upper_bound[0], self.resolution),
            torch.linspace(self.lower_bound[1], self.upper_bound[1], self.resolution),
            torch.linspace(self.lower_bound[2], self.upper_bound[2], self.resolution),
            indexing="ij",
        )
        grid_3d = torch.stack(grid, dim=-1).to("cuda")

        scene_points = Pointclouds(self.points.unsqueeze(0))
        grid_points = Pointclouds(grid_3d.reshape(-1, 3).unsqueeze(0))
        nn_dists, nn_idxs, _ = knn_points(
            oputil.convert_pointclouds_to_tensor(grid_points)[0],
            oputil.convert_pointclouds_to_tensor(scene_points)[0],
            K=1,
            return_nn=False,
        )

        nn_dists, nn_idxs = nn_dists.squeeze().sqrt(), nn_idxs.squeeze()

        self.nn_idxs_grid = nn_idxs.reshape(
            self.resolution, self.resolution, self.resolution
        )
        self.nn_dists_grid = nn_dists.reshape(
            self.resolution, self.resolution, self.resolution
        )

    def query(self, query_points: torch.Tensor, scale: float):
        scene_points = Pointclouds(self.points.unsqueeze(0) * scale)
        query_pts = Pointclouds(query_points.unsqueeze(0))
        nn_dists, nn_idxs, _ = knn_points(
            oputil.convert_pointclouds_to_tensor(query_pts)[0],
            oputil.convert_pointclouds_to_tensor(scene_points)[0],
            K=1,
            return_nn=False,
        )
        _, nn_idxs = nn_dists.squeeze().sqrt(), nn_idxs.squeeze()

        return nn_idxs

    def forward(self, query_points: torch.Tensor, scale, precompute=True):
        """
        Computes the Euclidean distances between each query point and its nearest scene point in the scaled space.
        Args:
            query_points (torch.Tensor): Tensor of shape (N, 3) with 3D coordinates of the query points.
            scale (float): A scaling factor applied to the scene points.
            precompute (bool, optional): If True, uses a precomputed grid lookup for nearest neighbors.
                                         If False, computes the nearest neighbor using k-NN. Default is True.

        Returns:
            nearest_neighbor_dists (torch.Tensor): Tensor of shape (N,) with the Euclidean distances between each
                                                    query point and its nearest scene point in the scaled space.
            nn_idxs (torch.Tensor): Tensor of shape (N,) with the indices of the nearest scene points.
        """

        if not precompute:
            nn_idxs = self.query(query_points, scale)
        else:
            knn_query_pts_scaled = query_points * (1 / scale)
            grid_indices = torch.round(
                (knn_query_pts_scaled - self.lower_bound)
                / (self.upper_bound - self.lower_bound)
                * (self.resolution - 1)
            ).long()

            if torch.all(grid_indices >= 0) and torch.all(
                grid_indices < self.resolution
            ):
                # Get indices
                nn_idxs = self.nn_idxs_grid[
                    grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]
                ]
            else:
                # If any query point is outside the grid, compute nearest neighbors using k-NN
                nn_idxs = self.query(query_points, scale)

        # For grid lookup - un-scale query points
        scene_points_scaled = self.points * scale

        # Computing distances in scaled space
        nearest_neighbor_dists = scene_points_scaled[nn_idxs] - query_points
        nearest_neighbor_dists = torch.norm(nearest_neighbor_dists, dim=-1)

        return nearest_neighbor_dists, nn_idxs


def rotmat_to_rot6d(R):
    """
    Converts a batch of 3x3 rotation matrices into a 6D continuous rotation representation.
    This function takes as input a tensor of rotation matrices and returns a tensor
    where each 3x3 rotation matrix is represented by its first two columns, concatenated
    to form a 6-dimensional vector. This 6D representation is often used to facilitate
    learning in neural networks by avoiding discontinuities present in other representations
    of rotations.
    Parameters:
        R (torch.Tensor): A tensor of shape (..., 3, 3) containing rotation matrices.
                          The tensor can have additional dimensions for batching (e.g., (B, N, 3, 3)).
    Returns:
        torch.Tensor: A tensor of shape (..., 6) where each 6-dimensional vector is constructed
                      by concatenating the first two columns of the corresponding 3x3 rotation matrix.
    """

    return torch.concatenate([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_rotmat(rots: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) or (B, N, 6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """

    if rots.dim() == 3:
        # we have (B, N, 6), reshape to (B*N, 6)
        x = rots.view(-1, 6)
    else:
        x = rots

    x = x.reshape(-1, 2, 3).contiguous()
    a1 = x[:, 0, :]
    a2 = x[:, 1, :]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rotmats = torch.stack((b1, b2, b3), dim=-1)

    if rots.dim() == 3:
        rotmats = rotmats.view(rots.shape[0], rots.shape[1], 3, 3)

    return rotmats


def batch_rot2aa(Rs):
    """
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);

        out_theta = std::acos(c);

        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """
    cos = 0.5 * (torch.stack([torch.trace(x) for x in Rs]) - 1)
    cos = torch.clamp(cos, -1, 1)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)


def rotmat_to_aa(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = (
            torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device)
            .reshape(1, 3, 1)
            .expand(rot_mat.shape[0], -1, -1)
        )
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix))
        )

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape
            )
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape
            )
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1  # noqa
        + t2_rep * mask_c2
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def project_points_torch(points, cam_intr):
    """
    Projects 3D points in camera coordinates to 2D image coordinates using torch.

    Args:
        points (torch.Tensor): A tensor of shape (N, 3) containing 3D points.
        cam_intr (torch.Tensor): A 3x3 camera intrinsic matrix.

    Returns:
        torch.Tensor: A tensor of shape (N, 2) with the 2D image coordinates.
    """
    points_h = points / points[:, 2:3]
    projected_h = cam_intr @ points_h.T
    return projected_h[:2].T


def project_to_floor_plane(points_3d, plane_points, plane_normal):
    """
    Project 3D points onto a floor plane or XZ plane if no floor is detected.
    Args:
        points_3d (torch.Tensor): 3D points to project, shape (N, 3)
        plane_points (torch.Tensor): Points defining the floor plane
        plane_normal (torch.Tensor): Normal vector of the floor plane, shape (3,)
    Returns:
        torch.Tensor: Projected points onto the floor plane, shape (N, 3)

    """

    if plane_points.shape[0] == 0:
        # No floor plane detected, project to XZ plane (set Y to 0)
        projected_points = points_3d.clone()
        projected_points[:, 1] = 0
        return projected_points

    # Project to detected floor plane
    floor_origin = plane_points[0]

    # normalize the plane normal
    unit_normal = plane_normal / plane_normal.norm(p=2)

    # compute vector from a point on the plane to each 3D point
    vec = points_3d - floor_origin.unsqueeze(0)  # shape (N, 3)
    # distance along the normal for each point
    dist = (vec * unit_normal.unsqueeze(0)).sum(dim=1, keepdim=True)  # shape (N, 1)
    # project onto plane
    projected_points = points_3d - dist * unit_normal.unsqueeze(0)

    return projected_points


def pointmap_to_intrinsics(pts3d: torch.Tensor):
    """
    Compute the intrinsic matrix from 3D points in camera coordinates.
    This function computes the intrinsic matrix from 3D points in camera coordinates.
    The 3D points are assumed to be in the camera coordinate system, where the camera is
    located at the origin and the camera looks along the positive z-axis.
    Parameters:
        pts3d (torch.Tensor): A tensor of shape (H, W, 3) containing 3D points in camera coordinates.
    Returns:
        torch.Tensor: A tensor of shape (3, 3) representing the intrinsic matrix.
    """

    H, W, _ = pts3d.shape
    # create image coordinate grid
    u = torch.arange(W, device=pts3d.device).view(1, W).expand(H, W).float()
    v = torch.arange(H, device=pts3d.device).view(H, 1).expand(H, W).float()

    # assume the principal point is at the center of the image
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # extract 3D coordinates
    x = pts3d[..., 0]
    y = pts3d[..., 1]
    z = pts3d[..., 2]

    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)

    eps = 1e-6
    # compute focal estimates from x-direction: x = (u - cx)*z/f  =>  f = (u - cx)*z/x
    mask_x = (torch.abs(u - cx) > eps) & (torch.abs(x) > eps) & valid
    f_x = torch.abs((u[mask_x] - cx) * z[mask_x] / x[mask_x])

    # compute focal estimates from y-direction: y = (v - cy)*z/f  =>  f = (v - cy)*z/y
    mask_y = (torch.abs(v - cy) > eps) & (torch.abs(y) > eps) & valid
    f_y = torch.abs((v[mask_y] - cy) * z[mask_y] / y[mask_y])

    # also drop any inf/NaN that might still sneak in
    f_x = f_x[torch.isfinite(f_x)]
    f_y = f_y[torch.isfinite(f_y)]

    # take the median value from valid estimates for robustness
    if f_x.numel() and f_y.numel():
        f = 0.5 * (f_x.median() + f_y.median())
    elif f_x.numel():
        f = f_x.median()
    elif f_y.numel():
        f = f_y.median()
    else:
        # fallback: if all points are degenerate, use a default focal length of 1.0
        f = torch.tensor(1.0, device=pts3d.device)

    # construct the intrinsic matrix
    intrinsics = torch.tensor(
        [[f, 0, cx], [0, f, cy], [0, 0, 1]], device=pts3d.device, dtype=pts3d.dtype
    )

    return intrinsics


def apply_transformation(points, transform):
    """
    Apply a 4x4 transformation matrix to a set of 3D points.

    Parameters:
    points (np.ndarray): An array of shape (N, 3) containing 3D points.
    transform (np.ndarray): A 4x4 transformation matrix.

    Returns:
    np.ndarray: The transformed 3D points of shape (N, 3).
    """
    # Convert points to homogeneous coordinates
    num_points = points.shape[0]
    homog_points = np.hstack([points, np.ones((num_points, 1))])

    # Apply transformation
    transformed = homog_points @ transform.T

    # Convert back to 3D
    return transformed[:, :3]
