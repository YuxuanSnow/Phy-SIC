"""Shared machinery for turning raw pipeline outputs into the released files.

The raw outputs carry the full reconstructed scene (a few million points per
frame) plus renders and meshes. The evaluation only ever looks at the scene
through one question -- *what is the nearest scene point to each body vertex,
and which way does its normal face* -- so a small subset of the cloud is enough
to reproduce every contact number exactly. That subset is what we release.
"""

import numpy as np
import torch


def scene_cloud_from_predictions(final_predictions):
    """Scene points + normals exactly as the pipeline feeds them to the contact test.

    The reconstructed points and the fitted floor plane are concatenated, and
    the points are brought to metric scale.
    """
    scale = final_predictions['scale']
    scene_points = np.concatenate([final_predictions['pts3d'] * scale,
                                   final_predictions['plane_points'] * scale], axis=0)
    scene_normals = np.concatenate(
        [final_predictions['normals'],
         np.tile(final_predictions['plane_normal'], (final_predictions['plane_points'].shape[0], 1))],
        axis=0)
    return scene_points, scene_normals


def nearest_neighbour_indices(scene_points, human_vertices, device='cuda'):
    """Per-vertex nearest scene point, using the same kNN the pipeline uses."""
    from pytorch3d.ops import knn_points

    scene_t = torch.as_tensor(scene_points, dtype=torch.float32, device=device)[None]
    verts_t = torch.as_tensor(human_vertices, dtype=torch.float32, device=device)[None]
    _, idx, _ = knn_points(verts_t, scene_t, K=1, return_nn=False)
    return idx.squeeze(0).squeeze(-1).cpu().numpy()


def contact_relevant_subset(scene_points, human_vertices, nn_idx, radius, device='cuda'):
    """Indices of the scene points the contact metrics can possibly touch.

    The union of (a) the nearest neighbour of every body vertex -- which fully
    determines the contact mask and the vertex-to-scene vectors, at any distance
    threshold -- and (b) every point within ``radius`` of the body, kept so the
    released cloud still carries the local scene geometry.
    """
    keep = set(nn_idx.tolist())

    if radius > 0:
        lo = human_vertices.min(axis=0) - radius
        hi = human_vertices.max(axis=0) + radius
        in_box = np.where(np.all((scene_points >= lo) & (scene_points <= hi), axis=1))[0]

        if in_box.size:
            candidates = torch.as_tensor(scene_points[in_box], dtype=torch.float32, device=device)
            verts_t = torch.as_tensor(human_vertices, dtype=torch.float32, device=device)
            near = torch.zeros(candidates.shape[0], dtype=torch.bool, device=device)
            for start in range(0, candidates.shape[0], 4096):
                stop = min(start + 4096, candidates.shape[0])
                d2 = ((candidates[start:stop, None, :] - verts_t[None, :, :]) ** 2).sum(-1)
                near[start:stop] = d2.min(dim=1).values <= radius ** 2
            keep.update(in_box[near.cpu().numpy()].tolist())

    return np.array(sorted(keep), dtype=np.int64)


def check_subset_is_faithful(vecs_ref, vecs_subset, mask_ref, mask_subset, frame_path):
    """Verify the stripped cloud reproduces the full cloud's contact result.

    Returns the number of vertices whose nearest neighbour was re-decided between
    two equidistant scene points: the released brute-force kNN may break a
    float32 tie the other way than the pipeline's kNN did, which moves the
    vertex-to-scene vector but never the distance, and never the contact bit.
    """
    assert np.array_equal(mask_ref, mask_subset), \
        f'contact mask changed after stripping: {frame_path}'
    assert np.allclose(np.linalg.norm(vecs_ref, axis=1), np.linalg.norm(vecs_subset, axis=1), atol=1e-5), \
        f'nearest-neighbour distance changed after stripping: {frame_path}'
    return int((np.linalg.norm(vecs_ref - vecs_subset, axis=1) > 1e-5).sum())
