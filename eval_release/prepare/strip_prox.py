#!/usr/bin/env python
"""Strip a PROX-quantitative run down to the minimal per-frame files we release.

Reads the raw pipeline outputs (``scene_data.pkl`` / ``scene_data_final.pkl``,
tens of MB per frame, plus renders and meshes) and writes one small ``.npz`` per
frame holding only what the metrics consume. Nothing derived from the PROX
imagery survives into the output.

Two things are computed here rather than at evaluation time, because they need
either the input image or the full-resolution scene cloud:

* the body-vertex **visibility mask**, from Grounded-SAM person segmentation of
  the PROX colour frame -- this is the only place a PROX image is read, and only
  a 10475-bool array comes out of it;
* the **contact-relevant subset** of the reconstructed scene cloud, verified
  per frame to reproduce the full cloud's contact result exactly.

The released evaluation (``evaluate.py``) does not need it.

Reference only: this documents how the released files were made. It needs the
raw pipeline outputs and a version of ``human.py`` whose contact function
takes ``return_vecs``, neither of which is in the public repo, so it will not
run from a fresh clone.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.body import load_smplx_layer, pose_prediction  # noqa: E402
from lib.contact import contact_from_pointcloud, point_visibility_mask  # noqa: E402
from lib.geometry import project_points_torch  # noqa: E402
from lib.release_io import BODY_PARAM_KEYS, save_frame  # noqa: E402
from prepare.stripping import (check_subset_is_faithful, contact_relevant_subset,  # noqa: E402
                               nearest_neighbour_indices, scene_cloud_from_predictions)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='pipeline output directory, laid out as <recording>/<frame>/')
    parser.add_argument('--prox-root', type=Path, required=True,
                        help='PROX quantitative root')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='default: release_data/ for the slice, release_data_full/ for --full-cloud')
    parser.add_argument('--smplx-model', type=Path,
                        default=REPO_ROOT / 'data/body_models/smplx/SMPLX_NEUTRAL.npz')
    parser.add_argument('--radius', type=float, default=0.05,
                        help='keep every scene point within this distance (m) of the body, '
                             'on top of the per-vertex nearest neighbours')
    parser.add_argument('--full-cloud', action='store_true',
                        help='store the complete reconstructed scene cloud instead of the '
                             'contact-relevant slice: much larger, and the evaluation gives '
                             'identical numbers either way')
    parser.add_argument('--limit', type=int, default=None, help='only process the first N frames')

    args = parser.parse_args()
    if args.out_dir is None:
        root = 'release_data_full' if args.full_cloud else 'release_data'
        args.out_dir = Path(__file__).resolve().parents[1] / root / 'prox'
    return args


def person_visibility_mask(image, vertices, K, device='cuda'):
    """Body vertices that project inside the segmented person in the input frame."""
    from image_helpers import get_mask_and_bbox

    masks, _, _, _ = get_mask_and_bbox(image)
    person_mask = np.stack(masks, axis=0).max(axis=0)

    verts_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    vertices_2d = project_points_torch(verts_t, torch.as_tensor(K, dtype=torch.float32, device=device))
    return point_visibility_mask(person_mask, vertices_2d).cpu().numpy()


def main():
    args = parse_args()
    prox_root = args.prox_root

    from human import get_contact_mask_from_pointcloud_pt  # reference implementation
    from image_helpers import load_gsam

    load_gsam()
    smplx_layer = load_smplx_layer(args.smplx_model)

    frame_paths = sorted(p for p in args.run_dir.glob('*/*') if p.is_dir())
    if args.limit:
        frame_paths = frame_paths[:args.limit]
    print(f'{len(frame_paths)} frames in {args.run_dir}')

    kept_points, n_tie_breaks = [], 0
    for frame_path in tqdm(frame_paths):
        recording, frame = frame_path.parts[-2], frame_path.parts[-1]

        with open(frame_path / 'scene_data.pkl', 'rb') as handle:
            initial_predictions = pickle.load(handle)
        with open(frame_path / 'scene_data_final.pkl', 'rb') as handle:
            final_predictions = pickle.load(handle)

        vertices, _ = pose_prediction(smplx_layer, final_predictions['body_params'],
                                      final_predictions['cam_trans'])

        scene_points, scene_normals = scene_cloud_from_predictions(final_predictions)

        # Reference contact on the full cloud, from the pipeline's own function.
        mask_ref, vecs_ref = get_contact_mask_from_pointcloud_pt(
            scene_points, scene_normals, vertices, return_vecs=True)
        vecs_ref = vecs_ref.cpu().numpy()

        if args.full_cloud:
            keep = np.arange(len(scene_points))
        else:
            nn_idx = nearest_neighbour_indices(scene_points, vertices)
            keep = contact_relevant_subset(scene_points, vertices, nn_idx, args.radius)
        kept_points.append(len(keep))

        subset_points = scene_points[keep]
        subset_normals = scene_normals[keep]

        mask_sub, vecs_sub = contact_from_pointcloud(subset_points, subset_normals, vertices)
        n_tie_breaks += check_subset_is_faithful(vecs_ref, vecs_sub, mask_ref, mask_sub, frame_path)

        # PROX quantitative recordings are mirrored; the pipeline consumed the flipped frame.
        image = Image.open(prox_root / f'recordings/{recording}/Color/{frame}.jpg')
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        visibility_mask = person_visibility_mask(image, vertices, final_predictions['K'])

        save_frame(
            args.out_dir / recording / f'{frame}.npz',
            init_params={k: initial_predictions['body_params_init'][k] for k in BODY_PARAM_KEYS},
            init_cam_trans=initial_predictions['cam_trans_init'],
            final_params={k: final_predictions['body_params'][k] for k in BODY_PARAM_KEYS},
            final_cam_trans=final_predictions['cam_trans'],
            K=final_predictions['K'],
            contact_deco=final_predictions['contact_vertices'][0],
            scene_points=subset_points,
            scene_normals=subset_normals,
            contact_pred_reference=mask_ref,
            visibility_mask=visibility_mask,
        )

    size_mb = sum(p.stat().st_size for p in args.out_dir.rglob('*.npz')) / 1e6
    print(f'wrote {len(frame_paths)} frames to {args.out_dir} ({size_mb:.1f} MB)')
    print(f'scene points kept per frame: mean {np.mean(kept_points):.0f}, max {np.max(kept_points)}')
    print(f'equidistant nearest-neighbour tie-breaks: {n_tie_breaks} of {len(frame_paths) * 10475} vertices')


if __name__ == '__main__':
    main()
