#!/usr/bin/env python
"""Strip a RICH run down to the minimal per-frame files we release.

Same idea as ``strip_prox.py``: keep the SMPL-X predictions, the intrinsics, the
DECO contact prediction and the contact-relevant slice of the reconstructed
scene cloud; drop the renders, the depth maps and the full-resolution geometry.
RICH scores contact over every body vertex, so no visibility mask is stored.

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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.body import load_smplx_layer, pose_prediction  # noqa: E402
from lib.contact import contact_from_pointcloud  # noqa: E402
from lib.release_io import BODY_PARAM_KEYS, save_frame  # noqa: E402
from prepare.stripping import (check_subset_is_faithful, contact_relevant_subset,  # noqa: E402
                               nearest_neighbour_indices, scene_cloud_from_predictions)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='pipeline output directory, one folder per frame')
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
        args.out_dir = Path(__file__).resolve().parents[1] / root / 'rich'
    return args


def main():
    args = parse_args()

    from human import get_contact_mask_from_pointcloud_pt  # reference implementation

    smplx_layer = load_smplx_layer(args.smplx_model)

    frame_paths = sorted(p for p in args.run_dir.glob('*_*_*') if p.is_dir())
    if args.limit:
        frame_paths = frame_paths[:args.limit]
    print(f'{len(frame_paths)} frames in {args.run_dir}')

    kept_points, n_tie_breaks = [], 0
    for frame_path in tqdm(frame_paths):
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

        save_frame(
            args.out_dir / f'{frame_path.name}.npz',
            init_params={k: initial_predictions['body_params_init'][k] for k in BODY_PARAM_KEYS},
            init_cam_trans=initial_predictions['cam_trans_init'],
            final_params={k: final_predictions['body_params'][k] for k in BODY_PARAM_KEYS},
            final_cam_trans=final_predictions['cam_trans'],
            K=final_predictions['K'],
            contact_deco=final_predictions['contact_vertices'][0],
            scene_points=subset_points,
            scene_normals=subset_normals,
            contact_pred_reference=mask_ref,
        )

    size_mb = sum(p.stat().st_size for p in args.out_dir.rglob('*.npz')) / 1e6
    print(f'wrote {len(frame_paths)} frames to {args.out_dir} ({size_mb:.1f} MB)')
    print(f'scene points kept per frame: mean {np.mean(kept_points):.0f}, max {np.max(kept_points)}')
    print(f'equidistant nearest-neighbour tie-breaks: {n_tie_breaks} of {len(frame_paths) * 10475} vertices')


if __name__ == '__main__':
    main()
