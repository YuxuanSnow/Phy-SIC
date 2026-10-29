"""RICH evaluation of the released predictions, over the 100-frame sample
listed in ``assets/rich_sample_frames.txt``.

Ground truth used, all from RICH:
  ``sample/<seq>+<frame>_gt.ply``                     ground-truth body mesh
  ``calibration/<scene>/calibration/0<cam>.xml``      camera extrinsics
  ``scene/scan_calibration/<scene>/<scan>.ply``       scanned scene mesh

RICH's ground truth is a gendered SMPL-X body, while our predictions use the
neutral model, so the neutral SMPL-X model is re-fitted with ``smplfitter`` to
the ground-truth body mesh -- this is what the published numbers compare
against. Predictions are in camera coordinates; the ground-truth body and scene
scan are brought there with the camera extrinsics.
"""

import os
import pickle

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from .body import load_smplx_layer, pose_prediction
from .contact import build_mesh_scene, contact_from_mesh_scene, contact_from_pointcloud
from .geometry import aa_to_rotmat, apply_transformation
from .metrics import (contact_scores, mean_point_error, pa_mpjpe,
                      weighted_contact_vector_angle_deg, weighted_contact_vector_l2)
from .release_io import load_frame
from .rich_camera import world_to_camera_transform

# Which scan of a scene an activity was recorded against: a few RICH scenes were
# rescanned with different furniture in place.
ACTIVITY_TO_SCAN = {
    'cooking1': 'scan_table_camcoord.ply',
    'dips1': 'scan_camcoord.ply',
    'dips2': 'scan_camcoord.ply',
    'lunge1': 'scan_camcoord.ply',
    'lunge2': 'scan_camcoord.ply',
    'mountainclimber1': 'scan_camcoord.ply',
    'mountainclimber2': 'scan_camcoord.ply',
    'pushup1': 'scan_camcoord.ply',
    'pushup2': 'scan_camcoord.ply',
    'plankjack1': 'scan_yoga_scene_camcoord.ply',
    'sidebalancerun1': 'scan_yoga_scene_camcoord.ply',
}


def scan_key(meta, scan_suffix):
    """Which scene scan, seen from which camera, a frame needs.

    The same scan filename appears under several scenes, so the scene is part of
    the key.
    """
    scan = ACTIVITY_TO_SCAN[meta['activity']].replace('.ply', f'{scan_suffix}.ply')
    return meta['scene'], scan, meta['camera']


def load_scan(rich_root, key, mesh_cache=None):
    """Load a scene scan, ready to be transformed into a camera's frame.

    Parsing a RICH scan costs far more than transforming one, and the evaluated
    frames need only three distinct scans between them, so the parsed mesh is
    kept and copied per camera. Pass ``mesh_cache=None`` to reload every time.
    """
    path = rich_root / f'scene/scan_calibration/{key[0]}/{key[1]}'
    if mesh_cache is None:
        return trimesh.load(path)
    if path not in mesh_cache:
        mesh_cache[path] = trimesh.load(path)
    return mesh_cache[path].copy()


def parse_frame_name(name):
    """``Gym_010_cooking1+00032_03`` -> scene, activity, camera id, sequence stem."""
    parts = name.split('_')
    return {
        'scene': parts[0],
        'activity': parts[2].split('+')[0],
        'camera': parts[-1],
        'sequence': name[:-3],
    }


def fit_gt_body(fitter, smplx_layer, mesh_path, device):
    """Fit the neutral SMPL-X model to a RICH ground-truth body mesh.

    Returns (vertices, joints) in RICH world coordinates.
    """
    mesh = trimesh.load(mesh_path)
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).float().to(device).unsqueeze(0)

    fit = fitter.fit(vertices, num_iter=3, beta_regularizer=1)
    rotvecs = fit['pose_rotvecs']

    output = smplx_layer(
        betas=fit['shape_betas'][0].unsqueeze(0),
        global_orient=aa_to_rotmat(rotvecs[:, :3])[0].unsqueeze(0),
        body_pose=aa_to_rotmat(rotvecs[:, 3:66].reshape(-1, 3)).reshape(1, 21, 3, 3),
        left_hand_pose=aa_to_rotmat(rotvecs[:, 75:120].reshape(-1, 3)).reshape(1, 15, 3, 3),
        right_hand_pose=aa_to_rotmat(rotvecs[:, 120:165].reshape(-1, 3)).reshape(1, 15, 3, 3),
        transl=fit['trans'][0].unsqueeze(0),
    )
    return (output.vertices[0].detach().cpu().numpy(),
            output.joints[0].detach().cpu().numpy())


def evaluate_rich(rich_root, data_dir, data_root, contact_weights_path, scan_suffix='',
                  limit=None, no_scene_cache=False, device='cuda'):
    """Score the predictions under ``data_dir`` (one ``<frame>.npz`` each) against RICH."""
    # smplfitter resolves its body models relative to DATA_ROOT; point it at the same
    # models as everything else, whatever the environment says.
    os.environ['DATA_ROOT'] = str(data_root)
    from smplfitter.pt import BodyFitter, BodyModel

    smplx_layer = load_smplx_layer(data_root / 'body_models/smplx/SMPLX_NEUTRAL.npz',
                                   device=device)
    fitter = BodyFitter(BodyModel('smplx', 'neutral', num_betas=10).to(device)).to(device)

    with open(contact_weights_path, 'rb') as handle:
        contact_weights = pickle.load(handle)
    contact_vertices_mask = contact_weights['contact_vertices_mask']
    contact_vertices_weights = contact_weights['contact_vertices_weights']

    frame_paths = sorted(data_dir.glob('*.npz'))
    if not frame_paths:
        raise SystemExit(f'no prediction files under {data_dir}')
    if limit:
        frame_paths = frame_paths[:limit]
    print(f'{len(frame_paths)} frames from {data_dir}')

    # A RICH scan runs to millions of faces and building its ray-casting structure is the
    # slowest step here, so visit the frames grouped by the scan-and-camera they need and
    # keep just the current one. Results are written back into frame order, so the visiting
    # order leaves even the floating-point summation in the metrics untouched.
    visit_order = list(range(len(frame_paths)))
    if not no_scene_cache:
        visit_order.sort(key=lambda i: scan_key(parse_frame_name(frame_paths[i].stem), scan_suffix))

    def per_frame():
        return [None] * len(frame_paths)

    init_joints, final_joints, gt_joints = per_frame(), per_frame(), per_frame()
    init_verts, final_verts, gt_verts = per_frame(), per_frame(), per_frame()
    pred_contacts, deco_contacts, gt_contacts = per_frame(), per_frame(), per_frame()
    pred_vecs, gt_vecs = per_frame(), per_frame()
    mesh_cache = {}
    cached_key, scene_scan = None, None

    for slot in tqdm(visit_order):
        frame_path = frame_paths[slot]
        meta = parse_frame_name(frame_path.stem)
        prediction = load_frame(frame_path)

        verts_cam_init, joints_cam_init = pose_prediction(
            smplx_layer, prediction['init_params'], prediction['init_cam_trans'], device)
        verts_cam, joints_cam = pose_prediction(
            smplx_layer, prediction['final_params'], prediction['final_cam_trans'], device)

        world_to_cam = world_to_camera_transform(
            rich_root / f"calibration/{meta['scene']}/calibration/0{meta['camera']}.xml")

        gt_body_verts, gt_body_joints = fit_gt_body(
            fitter, smplx_layer, rich_root / f"sample/{meta['sequence']}_gt.ply", device)

        key = scan_key(meta, scan_suffix)
        if key != cached_key or no_scene_cache:
            scan = load_scan(rich_root, key, None if no_scene_cache else mesh_cache)
            scan.apply_transform(world_to_cam)
            scene_scan, cached_key = build_mesh_scene(scan), key

        # RICH predictions stay in the camera frame; the ground truth is brought to it.
        init_verts[slot] = verts_cam_init
        init_joints[slot] = joints_cam_init
        final_verts[slot] = verts_cam
        final_joints[slot] = joints_cam
        gt_verts[slot] = apply_transformation(gt_body_verts, world_to_cam)
        gt_joints[slot] = apply_transformation(gt_body_joints, world_to_cam)

        contact, vecs = contact_from_pointcloud(
            prediction['scene_points'], prediction['scene_normals'], verts_cam,
            device=device)
        assert np.array_equal(contact, prediction['contact_pred_reference']), (
            f'contact mask does not match the value recorded at release time: {frame_path}')

        gt_contact, gt_vec = contact_from_mesh_scene(scene_scan, gt_verts[slot])

        pred_contacts[slot] = contact
        deco_contacts[slot] = prediction['contact_deco']
        gt_contacts[slot] = gt_contact
        pred_vecs[slot] = vecs
        gt_vecs[slot] = gt_vec

    init_joints = np.stack(init_joints)
    final_joints = np.stack(final_joints)
    gt_joints = np.stack(gt_joints)
    init_verts = np.stack(init_verts)
    final_verts = np.stack(final_verts)
    gt_verts = np.stack(gt_verts)

    pose = {
        'initial': {'pa_mpjpe': pa_mpjpe(init_joints, gt_joints),
                    'mpjpe': mean_point_error(init_joints, gt_joints),
                    'mpvpe': mean_point_error(init_verts, gt_verts)},
        'ours': {'pa_mpjpe': pa_mpjpe(final_joints, gt_joints),
                 'mpjpe': mean_point_error(final_joints, gt_joints),
                 'mpvpe': mean_point_error(final_verts, gt_verts)},
    }

    gt_flat = np.concatenate(gt_contacts)
    contact = {'all': {'ours': contact_scores(gt_flat, np.concatenate(pred_contacts)),
                       'deco': contact_scores(gt_flat, np.concatenate(deco_contacts))}}

    vectors = {
        'v2s_l2_m': weighted_contact_vector_l2(
            pred_vecs, gt_vecs, contact_vertices_mask, contact_vertices_weights),
        'v2s_angle_deg': weighted_contact_vector_angle_deg(
            pred_vecs, gt_vecs, contact_vertices_mask, contact_vertices_weights),
    }

    return {'dataset': 'RICH', 'num_frames': len(frame_paths),
            'pose': pose, 'contact': contact, 'vectors': vectors}
