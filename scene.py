import torch
import trimesh

import numpy as np
from pytorch3d.ops import utils as oputil
from pytorch3d.ops import knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.structures import Meshes


from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PointLights,
)

device = torch.device("cuda")

# Set up camera, lights, and renderers with corrected orientation
R, T = look_at_view_transform(dist=1, elev=0, azim=180, device=device)
# rotate about the z-axis by 180 degrees
rot_z = torch.tensor(
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=device
).unsqueeze(0)

R = torch.matmul(R, rot_z)

T = torch.tensor([0, 0, 3], device=device).unsqueeze(0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
lights = PointLights(device=device, location=[[0, 0, 1]])

raster_settings = RasterizationSettings(image_size=512, faces_per_pixel=4)
mesh_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
)


def points_to_mesh(points_list, point_colors_list, masks_list=None, cum_vertices=0):
    """
    Converts a list of points and their colors into a mesh representation.
    Args:
        points (list): List of Points to be added to the mesh, of shape (H, W, 3).
        point_colors (list): List of Colors of the points, of shape (H, W, 3).
        masks (list): List of Masks for the points, of shape (H, W).
        cum_vertices (int): Cumulative number of vertices from previous meshes.
    Returns:
        all_verts (list): List of all vertices in the mesh.
        all_faces (list): List of all faces in the mesh.
        all_colours (list): List of all colors in the mesh.
    """

    if masks_list is None:
        masks_list = [None] * len(points_list)

    all_verts = []
    all_faces = []
    all_colours = []

    for points, point_colors, mask in zip(points_list, point_colors_list, masks_list):
        if len(point_colors.shape) == 2:
            raise ValueError(
                "point_colors must be of shape (H, W, 3) to make a mesh using neighbors"
            )

        if points.shape[0] == 0:
            continue

        if mask is None:
            mask = np.ones((point_colors.shape[0], point_colors.shape[1]), dtype=bool)

        pts_numpy = points.detach().cpu().numpy() if torch.is_tensor(points) else points
        mask = (
            mask & ~np.isnan(pts_numpy).any(axis=-1) & ~np.isinf(pts_numpy).any(axis=-1)
        )

        triangles = create_triangles(
            point_colors.shape[0], point_colors.shape[1], mask=mask
        )
        # points = points.reshape(-1, 3).to(device)
        # faces = torch.tensor(triangles + cum_vertices, dtype=torch.long).to(device)
        # colors = torch.from_numpy(point_colors).float().reshape(-1, 3).to(device) / 255.0

        # if mask is not None:
        #     points = points.reshape(-1, 3)[mask.reshape(-1)]

        points = points.reshape(-1, 3)
        faces = triangles + cum_vertices
        colors = point_colors.reshape(-1, 3) / 255.0

        if hasattr(points, "device"):
            points = points.to(device)
            faces = torch.tensor(faces, dtype=torch.long).to(device)
            colors = torch.from_numpy(colors).float().to(device)

        cum_vertices += points.shape[0]

        all_verts.append(points)
        all_faces.append(faces)
        all_colours.append(colors)

    return all_verts, all_faces, all_colours


def render_mesh_points(
    mesh_vertices, mesh_faces, mesh_vertex_colors, points_list, point_colors_list
):
    """
    Renders a mesh along with additional points and their colors. The points are converted into a mesh
    by creating a grid of triangles. The mesh and points are then rendered together using a mesh renderer.
    Args:
        mesh_vertices (torch.Tensor): Vertices of the mesh in 3D space.
        mesh_faces (torch.Tensor): Faces of the mesh vertices.
        mesh_vertex_colors (torch.Tensor): Colors of the mesh vertices in [0, 255]
        points (torch.Tensor): List of Points to be added to the mesh, of shape (H, W, 3).
        point_colors (numpy.ndarray): List of Colors of the points, of shape (H, W, 3).
    Returns:
        torch.Tensor: Rendered image of the mesh and points.
    """

    # Prepare a joint mesh
    verts = mesh_vertices.float().to(device)
    faces = mesh_faces.long().to(device)
    mesh_colors = (
        mesh_vertex_colors.float().to(device) / 255.0
    )  # torch.from_numpy(mesh.visual.vertex_colors[:, :3]).float() / 255.0

    all_verts = [verts]
    all_faces = [faces]
    all_colours = [mesh_colors]
    cum_vertices = verts.shape[0]

    # Convert points to mesh
    point_verts, point_faces, point_colours = points_to_mesh(
        points_list, point_colors_list, cum_vertices=cum_vertices
    )
    all_verts.extend(point_verts)
    all_faces.extend(point_faces)
    all_colours.extend(point_colours)

    all_verts = torch.cat(all_verts, dim=0)
    all_faces = torch.cat(all_faces, dim=0)
    all_colours = torch.cat(all_colours, dim=0)

    textures = TexturesVertex(all_colours[None])

    mesh_p3d = Meshes(verts=[all_verts], faces=[all_faces], textures=textures)

    # Render mesh and points together
    mesh_img = mesh_renderer(mesh_p3d)

    return mesh_img


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


def depth_to_points_pt(depth, K=None, R=None, t=None):
    """
    Converts a depth map to a point cloud in camera coordinates.
    This function takes a depth map along with an optional camera intrinsic matrix (K),
    and optionally applies a coordinate transformation using a rotation matrix (R) and a translation vector (t).
    It computes the 3D world points corresponding to the depth values by projecting pixel
    coordinates (augmented with a homogeneous coordinate of 1) back into 3D space using the inverse
    of the intrinsic matrix. An optional coordinate-to-world (c2w) transformation is then applied
    using R and t.
    Parameters:
        depth (torch.Tensor): A depth map tensor of shape (H, W) or (B, H, W), where H and W are the height and width.
                              Each entry represents the depth value at that pixel.
        K (torch.Tensor, optional): The camera intrinsic matrix of shape (3, 3) or (B, 3, 3). If provided, the inverse of
                                    this matrix is used to map pixel coordinates to the normalized camera coordinate system.
                                    Default is None.
        R (torch.Tensor, optional): A rotation matrix of shape (3, 3) used for an optional coordinate transformation.
                                    If None, the identity matrix is used.
        t (torch.Tensor, optional): A translation vector of shape (3,) used for an optional coordinate transformation.
                                    If None, a zero vector is used.
    Returns:
        torch.Tensor: A tensor of 3D points of shape (B, H, W, 3) representing the points in the camera (or world) coordinate system.
                      For each pixel, the corresponding 3D coordinates are computed from the depth and intrinsic parameters.
    Notes:
        - The function creates a grid of pixel coordinates and converts them to homogeneous coordinates.
        - If the intrinsic matrix K is provided as a single 3x3 matrix without a batch dimension, it is unsqueezed to match the batch size.
        - The function performs broadcasting where necessary to match dimensions when multiplying tensors.
        - After converting depth values to 3D points in the normalized camera coordinate system,
          they are scaled by the depth and optionally transformed by the provided R and t.
    """

    if R is None:
        R = torch.eye(3, device=depth.device)
    if t is None:
        t = torch.zeros(3, device=depth.device)

    Kinv = torch.inverse(K)

    h, w = depth.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device), torch.arange(w, device=depth.device)
    )
    coords = (
        torch.stack((x, y, torch.ones_like(x)), dim=-1)
        .float()
        .unsqueeze(0)
        .unsqueeze(-1)
    )  # (1, H, W, 3, 1)

    if len(Kinv.shape) == 2:
        # if single Kinv, add batch dimension
        Kinv = Kinv.unsqueeze(0)

    # (B, H, W, 1, 1) is the depth dimension, (B, 3, 3) is the intrinsics matrix
    # we need to broadcast the intrinsics matrix to the depth dimension, broadcasting will take care of the rest
    Kinv = Kinv.unsqueeze(1).unsqueeze(1)

    # Now, Kinv @ coords gives us the 3D points in the normalized camera coordinate system
    # To convert that into the camera coordinate system, we need to multiply by the depth
    # (B, H, W, 1, 1) * (B, 1, 1, 3, 3)  = (B, H, W, 3, 3) @ (1, H, W, 3, 1)  = (B, H, W, 3, 1)
    points = depth.unsqueeze(-1).unsqueeze(-1) * Kinv @ coords
    # optional c2w transformation
    points = R.unsqueeze(0) @ points + t.view(1, 1, 3, 1)
    # (B, H, W, 3, 1) -> (B, H, W, 3)
    return points[..., 0]


def get_rays_from_intrinsics(K, image_shape):
    """
    Generate rays from camera intrinsics.
    Parameters:
    K (numpy.ndarray): The camera intrinsic matrix of shape (3, 3).
    image_shape (tuple): The shape of the image (height, width).
    Returns:
    tuple: A tuple containing:
        - rays_o (numpy.ndarray): The origins of the rays, with shape (height, width, 3).
        - rays_v (numpy.ndarray): The directions of the rays, with shape (height, width, 3).

    """
    tx = np.linspace(0, image_shape[1] - 1, image_shape[1])  # pixel x coordinates
    # pixel y coordinates
    ty = np.linspace(0, image_shape[0] - 1, image_shape[0])
    pixels_x, pixels_y = np.meshgrid(tx, ty)
    p = np.stack(
        [pixels_x, pixels_y, np.ones_like(pixels_y)], axis=-1
    )  # pixels in homogeneous coordinates
    # apply inverse of intrinsic matrix
    p = np.einsum("ij,mnj->mni", np.linalg.inv(K), p)
    # normalize the points and make vectors of rays
    rays_v = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)
    # origin of the rays is the camera position, we kept it at -2 in the z-axis
    rays_o = np.broadcast_to(np.zeros((3,)), rays_v.shape)
    return rays_o, rays_v


def create_triangles(h, w, mask=None):
    """
    Reference: https://github.com/google-research/google-research/blob/e96197de06613f1b027d20328e06d69829fa5a89/infinite_nature/render_utils.py#L68
    Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles


def get_ray_mesh_intersections(ray_origins, ray_directions, mesh):
    ray_origins = ray_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    return mesh.ray.intersects_id(ray_origins, ray_directions)


def scale_points_to_human_mesh(mesh, pts3d, mask, K, threshold=0.80):
    """
    Scales the depth map to fit the human mesh using a binary search algorithm.
    Uses ray-mesh intersection to check if the points in the scene are behind the human mesh.

    Args:
    mesh (trimesh.Trimesh): The human mesh.
    pts3d (numpy.ndarray): The depth points (unprojected), of shape (H, W, 3).
    mask (numpy.ndarray): The mask of the human in the depth map. This needs to be precise, without dilations since we are checking for points behind the human.
    K (numpy.ndarray): The camera intrinsic matrix.

    Returns:
    float: The optimal scale factor for the scene to fit the human mesh.
    """

    # do a binary search to find the optimal scale factor for the scene to fit the human
    # since the human will be decently scaled, due to size constraints on the human
    rays_o, rays_v = get_rays_from_intrinsics(K, pts3d.shape[:2])
    triangles = create_triangles(pts3d.shape[0], pts3d.shape[1], mask=mask)
    l = 1.0
    r = 5
    scene_mask = mask > 128
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    while r - l > 0.01:
        mid = (l + r) / 2
        scaled_pts3d = pts3d * mid
        scaled_pts3d = scaled_pts3d[scene_mask].reshape(-1, 3)

        intersect_tris, intersect_rays, intersect_locations = intersector.intersects_id(
            rays_o[scene_mask].reshape(-1, 3),
            rays_v[scene_mask].reshape(-1, 3),
            multiple_hits=True,
            return_locations=True,
        )
        intersect_distances = np.linalg.norm(intersect_locations, axis=-1)
        max_ray_distances = np.zeros(scene_mask.sum())

        np.maximum.at(max_ray_distances, intersect_rays, intersect_distances)

        # now check for each point in pts3d if it is behind the human
        pts3d_distances = np.linalg.norm(scaled_pts3d, axis=-1)
        pts3d_behind = pts3d_distances > max_ray_distances

        if pts3d_behind.sum() / len(pts3d_behind) > threshold:
            # the scene is fine
            r = mid
        else:
            # the scene is too big
            l = mid

    return l


def compute_point_normals(pts3d):
    """
    Compute per-point normals for a 3D point grid.
    This function computes the normal vector at each point in a 2D grid of 3D points.
    It uses neighboring points to estimate the surface normal by taking cross-products
    of vectors formed between the point of interest and its adjacent neighbors. Four
    cross-products are computed using different pairs of immediate neighbors (top, right,
    bottom, left), normalized, and then averaged to obtain a robust normal estimate.
    Parameters:
        pts3d (torch.Tensor): A tensor of shape (H, W, 3) containing 3D coordinates for each point.
                              It can be on CPU or GPU.
    Returns:
        torch.Tensor: A tensor of shape (H, W, 3) containing the computed normalized normals.
                      For points where a valid normal cannot be computed (e.g., near boundaries or
                      where the geometry is degenerate), the normal vector is set to NaN.
    Notes:
        - The grid is padded with NaNs to facilitate easy neighbor access.
        - Normals are calculated from cross products of vectors defined between the central point
          and its four neighbors.
        - A small epsilon (1e-6) is used to avoid division by zero during normalization.

    """

    H, W, _ = pts3d.shape
    eps = 1e-6
    device = pts3d.device if hasattr(pts3d, "device") else None
    normals = torch.zeros((H, W, 3), dtype=pts3d.dtype, device=device)

    padded = torch.full(
        (H + 2, W + 2, 3), float("nan"), dtype=pts3d.dtype, device=device
    )
    padded[1:-1, 1:-1] = pts3d

    top = padded[:-2, 1:-1]
    right = padded[1:-1, 2:]
    bottom = padded[2:, 1:-1]
    left = padded[1:-1, :-2]

    p = pts3d

    v_top = top - p
    v_right = right - p
    v_bottom = bottom - p
    v_left = left - p

    cp0 = torch.cross(v_right, v_top, dim=2)
    cp1 = torch.cross(v_bottom, v_right, dim=2)
    cp2 = torch.cross(v_left, v_bottom, dim=2)
    cp3 = torch.cross(v_top, v_left, dim=2)

    norm0 = torch.norm(cp0, dim=2)
    norm1 = torch.norm(cp1, dim=2)
    norm2 = torch.norm(cp2, dim=2)
    norm3 = torch.norm(cp3, dim=2)

    def normalize(cp, norm):
        valid = norm > eps
        cp_norm = cp.clone()
        cp_norm[valid] = cp[valid] / norm[valid].unsqueeze(-1)
        cp_norm[~valid] = float("nan")
        return cp_norm

    cp0_n = normalize(cp0, norm0)
    cp1_n = normalize(cp1, norm1)
    cp2_n = normalize(cp2, norm2)
    cp3_n = normalize(cp3, norm3)

    cp_stack = torch.stack([cp0_n, cp1_n, cp2_n, cp3_n], dim=2)
    valid_mask = ~torch.isnan(cp_stack[..., 0])
    valid_count = torch.sum(valid_mask, dim=2, keepdim=True).float()

    cp_stack[torch.isnan(cp_stack)] = 0
    avg_norm = torch.sum(cp_stack, dim=2) / torch.where(
        valid_count == 0, torch.tensor(1.0, device=device), valid_count
    )

    norm_avg = torch.norm(avg_norm, dim=2)
    valid = norm_avg > eps
    normals[valid] = avg_norm[valid] / norm_avg[valid].unsqueeze(-1)
    return normals


def extend_floor_plane(pts3d, normals, mask_floor, steps=100):
    """
    Estimates and extends a floor plane based on 3D points and their corresponding normals.
    Args:
        pts3d (torch.Tensor): A tensor of shape (H, W, 3) containing 3D point coordinates.
        normals (torch.Tensor): A tensor of shape (H, W, 3) containing normal vectors for each point.
        mask_floor (numpy.ndarray): A boolean array indicating which points belong to the floor.
    Returns:
        plane_points (numpy.ndarray): A grid of points representing the extended floor plane.
        plane_normal (numpy.ndarray): The normal vector of the fitted plane.
        plane_d (float): The plane's intercept term (D) in the plane equation.
    """

    if not any(mask_floor.flatten()):
        # return empty tensors if no floor points are found
        return (
            torch.zeros(0, 3, device=pts3d.device),
            torch.zeros(3, device=pts3d.device),
            0.0,
        )

    # Move floor mask to GPU
    floor_mask = torch.from_numpy(mask_floor).bool().to(pts3d.device)
    # Gather floor points
    pts_floor = pts3d[floor_mask].float()
    valid_mask = torch.isfinite(pts3d).all(dim=-1)  # Shape: (H, W)
    pts3d = pts3d[
        valid_mask
    ]  # Shape: (N_valid, 3)            # remove NaN/Inf points from pts3d --> can cause issues
    print(f"pts3d shape: {pts3d.shape}, pts_floor shape: {pts_floor.shape}")
    norms_floor = normals[floor_mask].float()

    # RANSAC in PyTorch, but sample 20% of the floor points at a time
    best_normal, best_d, best_inliers = None, None, -1
    sample_size = max(1, int(0.2 * pts_floor.size(0)))  # pick 20% of floor points
    for _ in range(20):
        idx = torch.randint(pts_floor.size(0), (sample_size,), device=pts_floor.device)
        sample_points = pts_floor[idx]
        center = sample_points.mean(dim=0)
        # Fit plane using SVD on the sampled points
        _, _, V = torch.pca_lowrank(sample_points - center, q=3)
        n = V[:, -1]
        # Align plane normal with average of chosen points' normals
        avg_normal = norms_floor[idx].mean(dim=0)
        if torch.dot(n, avg_normal) < 0:
            n = -n
        d_val = -torch.dot(n, center)
        dist = torch.abs(pts_floor @ n + d_val)
        inliers = (dist < 0.01).sum().item()
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal, best_d = n, d_val

    # Refine using SVD on inliers
    diff = torch.abs(pts_floor @ best_normal + best_d)
    inlier_pts = pts_floor[diff < 0.01]
    inlier_normals = norms_floor[diff < 0.01]
    center = inlier_pts.mean(dim=0)
    _, _, V = torch.pca_lowrank(inlier_pts - center, q=3)
    plane_normal = V[:, -1]
    avg_inlier_normal = inlier_normals.mean(dim=0)
    if torch.dot(plane_normal, avg_inlier_normal) < 0:
        plane_normal = -plane_normal
    plane_d = -torch.dot(plane_normal, center)

    # Extend plane with truncated range
    x_vals = pts3d[..., 0].reshape(-1)
    z_vals = pts3d[..., 2].reshape(-1)
    x_vals_sorted, _ = torch.sort(x_vals)
    z_vals_sorted, _ = torch.sort(z_vals)

    print(x_vals, z_vals)

    idx_lower_x = int(0.02 * x_vals_sorted.numel())
    idx_upper_x = int(0.98 * x_vals_sorted.numel())
    x_min = x_vals_sorted[idx_lower_x]
    x_max = x_vals_sorted[idx_upper_x]

    idx_lower_z = int(0.02 * z_vals_sorted.numel())
    idx_upper_z = int(0.98 * z_vals_sorted.numel())
    z_min = z_vals_sorted[idx_lower_z]
    z_max = z_vals_sorted[idx_upper_z]

    xs = torch.linspace(x_min, x_max, steps, device=pts3d.device)
    zs = torch.linspace(z_min, z_max, steps, device=pts3d.device)
    grid = torch.cartesian_prod(xs, zs)

    y_vals = (
        -(plane_d + plane_normal[0] * grid[:, 0] + plane_normal[2] * grid[:, 1])
        / plane_normal[1]
    )
    plane_torch = torch.stack([grid[:, 0], y_vals, grid[:, 1]], dim=-1)

    plane_points = plane_torch
    plane_normal = plane_normal
    plane_d = plane_d.item()

    return plane_points, plane_normal, plane_d

def compute_outlier_mask(pts3d, k=6, std=1.8):
    """
    Compute an outlier mask for a set of 3D points using k-nearest neighbors,
    and mark any point containing NaN or ±Inf as an outlier automatically.

    Args:
        pts3d (Tensor): Input tensor of shape (..., 3).
        k (int): Number of neighbors to use for average distance.
        std (float): Multiplier for the standard deviation to set the threshold.

    Returns:
        Tensor[bool]: Boolean mask of shape pts3d.shape[:-1], where True indicates an outlier.
    """
    # Flatten points to (N, 3)
    pts_flat = pts3d.reshape(-1, 3)

    # 1) Mark all rows with NaN or Inf as outliers
    finite_mask = torch.isfinite(pts_flat).all(dim=1)  # True = valid point
    outlier_mask = torch.zeros_like(finite_mask)  # initialize all False
    outlier_mask[~finite_mask] = True  # flag invalid pts

    # 2) Run k-NN on the remaining finite points
    if finite_mask.sum() > k + 1:
        good_pts = pts_flat[finite_mask]
        pcd = Pointclouds(good_pts.unsqueeze(0))
        pts_tensor = oputil.convert_pointclouds_to_tensor(pcd)[0]
        nn_dists, _, _ = knn_points(pts_tensor, pts_tensor, K=k)
        avg_nn = nn_dists.mean(dim=-1).view(-1)

        # compute mean and std in ASCII-safe names
        avg_mean = avg_nn.mean()
        avg_std = avg_nn.std()
        threshold = avg_mean + std * avg_std

        # flag finite points whose avg-NN distance exceeds threshold
        nn_outliers = avg_nn > threshold
        outlier_mask[finite_mask] = nn_outliers

    # reshape back to original grid
    return outlier_mask.reshape(pts3d.shape[:-1])


def get_normal_outlier_mask(pts3d, normals, k=6, threshold_degrees=60):
    pts3d_flat = pts3d.reshape(-1, 3)
    normals_flat = normals.reshape(-1, 3)

    pcd = Pointclouds(pts3d_flat.unsqueeze(0))
    nn_dists, nn_idx, _ = knn_points(
        oputil.convert_pointclouds_to_tensor(pcd)[0],
        oputil.convert_pointclouds_to_tensor(pcd)[0],
        K=k + 1,
    )

    # Squeeze out the batch dimension of nn_idx so it has shape [N, K]
    nn_idx = nn_idx.squeeze(0)

    # Exclude the point itself (index 0) and get the normals of the k neighbors
    neighbor_normals = normals_flat[nn_idx[:, 1:]]

    # Compute the average normal for each point's neighborhood and normalize it
    avg_normal = neighbor_normals.mean(dim=1)
    avg_normal = avg_normal / avg_normal.norm(dim=1, keepdim=True).clamp(min=1e-6)

    # Normalize the original normals
    point_normals = normals_flat / normals_flat.norm(dim=1, keepdim=True).clamp(
        min=1e-6
    )

    # Compute the angle in degrees between each point's normal and its neighborhood average normal
    dot = (point_normals * avg_normal).sum(dim=1).clamp(-1, 1)
    angles = torch.acos(dot) * (180.0 / torch.pi)

    # Mark points as outliers if the angle difference exceeds the threshold
    mask = angles > threshold_degrees

    # Reshape the mask to match the original pts3d shape excluding the last dimension
    mask = mask.reshape(normals.shape[:-1])
    return mask
