"""Contact estimation between a posed body mesh and the scene.

Two variants are used by the paper's evaluation:

* :func:`contact_from_pointcloud` -- our *prediction*: the scene is the point
  cloud (+ fitted floor plane) that the optimiser reconstructed from the image.
* :func:`contact_from_mesh` -- the *ground truth*: the scene is the dataset's
  scanned mesh, and the closest point is taken on the mesh surface.

Both use the same rule: a body vertex is in contact if it is within
``threshold`` of the scene, or if it sits *behind* the scene surface (negative
dot product with the surface normal) and closer than
``interpenetration_threshold``.
"""

import numpy as np
import torch


# Peak working set for one chunk of the brute-force search, in bytes. The chunk
# size is derived from this so that a 10k-point scene slice and a 3M-point full
# cloud both fit; it is a memory knob only and never changes the result.
_NN_CHUNK_BYTES = 512 << 20


def nearest_neighbour(scene_points: torch.Tensor, queries: torch.Tensor, chunk: int = None):
    """Brute-force 1-nearest-neighbour of ``queries`` (M, 3) in ``scene_points`` (N, 3).

    Chunked over the queries so that memory stays bounded whether the scene is a
    few thousand points or a few million. Distances are computed from explicit
    coordinate differences (not the matmul expansion) to keep float32 error at
    the level of the reference implementation.

    Returns:
        (dists, idx): ``(M,)`` euclidean distances and ``(M,)`` indices.
    """
    if chunk is None:
        chunk = max(1, min(queries.shape[0], _NN_CHUNK_BYTES // (scene_points.shape[0] * 12)))

    dists = torch.empty(queries.shape[0], device=queries.device, dtype=queries.dtype)
    idx = torch.empty(queries.shape[0], device=queries.device, dtype=torch.long)
    for start in range(0, queries.shape[0], chunk):
        stop = min(start + chunk, queries.shape[0])
        d2 = ((queries[start:stop, None, :] - scene_points[None, :, :]) ** 2).sum(-1)
        chunk_d2, chunk_idx = d2.min(dim=1)
        dists[start:stop] = chunk_d2.sqrt()
        idx[start:stop] = chunk_idx
    return dists, idx


def contact_from_pointcloud(scene_points, scene_normals, human_vertices,
                            threshold=0.02, interpenetration_threshold=0.15,
                            device='cuda'):
    """Contact of each body vertex against a scene *point cloud*.

    Args:
        scene_points: (N, 3) scene points, metric scale, camera frame.
        scene_normals: (N, 3) unit normals, one per scene point.
        human_vertices: (M, 3) posed body vertices in the same frame.

    Returns:
        (mask, vecs): ``(M,)`` bool contact mask and ``(M, 3)`` vectors from the
        nearest scene point to the body vertex.
    """
    scene_points_t = torch.as_tensor(scene_points, dtype=torch.float32, device=device)
    scene_normals_t = torch.as_tensor(scene_normals, dtype=torch.float32, device=device)
    human_vertices_t = torch.as_tensor(human_vertices, dtype=torch.float32, device=device)

    dists, idx = nearest_neighbour(scene_points_t, human_vertices_t)

    nearest_pts = scene_points_t[idx]
    nearest_normals = scene_normals_t[idx]

    directions = human_vertices_t - nearest_pts
    direction_normalized = directions / (torch.norm(directions, dim=1, keepdim=True) + 1e-6)
    dot_products = (direction_normalized * nearest_normals).sum(dim=1)

    contact_from_dot = (dot_products < 0) & (dists < interpenetration_threshold)
    mask = (dists <= threshold) | contact_from_dot

    return mask.cpu().numpy(), directions.cpu().numpy()


def build_mesh_scene(scene_mesh):
    """Build the ray-casting acceleration structure for a scene mesh.

    Split out from :func:`contact_from_mesh` so a scene that many frames share
    -- PROX's single scan, or one RICH scan seen from one camera -- is built once.
    """
    import open3d as o3d

    # Open3D needs plain contiguous arrays of its own dtypes here; handing it
    # trimesh's int64 tracked arrays segfaults.
    vertices = np.ascontiguousarray(scene_mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(scene_mesh.faces, dtype=np.int32)

    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                         o3d.utility.Vector3iVector(faces))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))
    return scene


def contact_from_mesh_scene(scene, human_vertices,
                            threshold=0.02, interpenetration_threshold=0.15):
    """Contact against a scene already prepared by :func:`build_mesh_scene`."""
    import open3d as o3d

    human_vertices = np.asarray(human_vertices, dtype=np.float32)
    ret = scene.compute_closest_points(o3d.core.Tensor.from_numpy(human_vertices))
    points = ret['points'].numpy()
    normals = ret['primitive_normals'].numpy()

    distances = np.linalg.norm(points - human_vertices, axis=1)
    directions = human_vertices - points
    directions_normalized = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6)
    dot_products = np.sum(directions_normalized * normals, axis=1)

    mask = (distances <= threshold) | ((dot_products < 0) & (distances < interpenetration_threshold))
    return mask, directions


def contact_from_mesh(scene_mesh, human_vertices,
                      threshold=0.02, interpenetration_threshold=0.15):
    """Contact of each body vertex against a scene *mesh* (ground-truth scan).

    Distances are to the mesh *surface*, so large triangles are handled
    correctly (a vertex-only nearest neighbour would miss them).

    Args:
        scene_mesh: ``trimesh.Trimesh`` scene scan, in the same frame as the body.
        human_vertices: (M, 3) posed body vertices.

    Returns:
        (mask, vecs): ``(M,)`` bool contact mask and ``(M, 3)`` vectors from the
        closest surface point to the body vertex.
    """
    return contact_from_mesh_scene(build_mesh_scene(scene_mesh), human_vertices,
                                   threshold, interpenetration_threshold)


def point_visibility_mask(image_mask: np.ndarray, points_2d: torch.Tensor) -> torch.Tensor:
    """True where a projected point falls inside ``image_mask`` (H, W) and in bounds."""
    height, width = image_mask.shape[:2]
    xs = torch.round(points_2d[..., 0]).long()
    ys = torch.round(points_2d[..., 1]).long()
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    mask_t = torch.from_numpy(image_mask).to(points_2d.device)
    in_mask = torch.zeros_like(valid, dtype=torch.bool)
    in_mask[valid] = (mask_t[ys[valid], xs[valid]] != 0)
    return in_mask
