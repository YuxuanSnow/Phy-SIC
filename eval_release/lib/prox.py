"""PROX-quantitative evaluation of the released predictions.

Ground truth used, all from the PROX *quantitative* release:
  ``fittings/mosh/<recording>/results/<frame>/000.pkl``  MoSh SMPL-X fit
  ``scenes/vicon.ply``                                   scanned scene mesh
  ``cam2world/vicon.json`` and ``vicon2scene.json``      frame alignment

Predictions live in the camera frame and are mapped to the scene frame with
``cam2world``; the MoSh fit lives in the Vicon frame and is mapped with
``vicon2scene``. Contact is evaluated in each mesh's own frame: ours against
the reconstructed scene cloud in camera coordinates, the ground truth against
the scanned mesh in scene coordinates.
"""

import json
import pickle

import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm

from .body import load_smplx_layer, pose_prediction
from .contact import build_mesh_scene, contact_from_mesh_scene, contact_from_pointcloud
from .geometry import apply_transformation
from .metrics import (contact_scores, mean_point_error, pa_mpjpe,
                      weighted_contact_vector_angle_deg, weighted_contact_vector_l2)
from .release_io import load_frame


def load_mosh_body(prox_root, recording, frame, gt_body_model, device):
    """Ground-truth body from the PROX MoSh fit, in the Vicon frame.

    The parameter file is fed to SMPL-X as-is, exactly as in the original
    evaluation. Note it stores the shape under ``beta`` rather than ``betas``,
    so SMPL-X falls back to its mean shape -- the published numbers were
    computed that way and this reproduces them.
    """
    path = prox_root / f'fittings/mosh/{recording}/results/{frame}/000.pkl'
    with open(path, 'rb') as handle:
        mosh = pickle.load(handle, encoding='latin1')

    output = gt_body_model(**{k: torch.tensor(v).float().to(device)
                              for k, v in mosh.items() if isinstance(v, np.ndarray)})
    return (output.vertices[0].detach().cpu().numpy(),
            output.joints[0].detach().cpu().numpy())


def evaluate_prox(prox_root, data_dir, smplx_dir, contact_weights_path,
                  limit=None, no_scene_cache=False, device='cuda'):
    """Score the predictions under ``data_dir`` (``<recording>/<frame>.npz``) against PROX."""
    smplx_layer = load_smplx_layer(smplx_dir / 'SMPLX_NEUTRAL.npz', device=device)
    gt_body_model = smplx.SMPLX(model_path=str(smplx_dir / 'SMPLX_NEUTRAL.npz'),
                                num_betas=10, num_pca_comps=12).to(device)

    # PROX evaluates every frame against the same scan, so its ray-casting structure is
    # built once. no_scene_cache rebuilds it per frame; the result is identical.
    scene_mesh = trimesh.load(prox_root / 'scenes/vicon.ply')
    scene_gt = None if no_scene_cache else build_mesh_scene(scene_mesh)
    cam2world = np.array(json.load(open(prox_root / 'cam2world/vicon.json')))
    vicon2scene = np.array(json.load(open(prox_root / 'vicon2scene.json')))

    with open(contact_weights_path, 'rb') as handle:
        contact_weights = pickle.load(handle)
    contact_vertices_mask = contact_weights['contact_vertices_mask']
    contact_vertices_weights = contact_weights['contact_vertices_weights']

    frame_paths = sorted(data_dir.glob('*/*.npz'))
    if not frame_paths:
        raise SystemExit(f'no prediction files under {data_dir}')
    if limit:
        frame_paths = frame_paths[:limit]
    print(f'{len(frame_paths)} frames from {data_dir}')

    init_joints, final_joints, gt_joints = [], [], []
    init_verts, final_verts, gt_verts = [], [], []
    pred_contacts, deco_contacts, gt_contacts, visibility = [], [], [], []
    pred_vecs, gt_vecs = [], []

    for frame_path in tqdm(frame_paths):
        recording, frame = frame_path.parts[-2], frame_path.stem
        prediction = load_frame(frame_path)

        verts_cam_init, joints_cam_init = pose_prediction(
            smplx_layer, prediction['init_params'], prediction['init_cam_trans'], device)
        verts_cam, joints_cam = pose_prediction(
            smplx_layer, prediction['final_params'], prediction['final_cam_trans'], device)

        # Ours is predicted in camera coordinates; the ground truth lives in the scene
        # frame, so bring both there before comparing.
        init_verts.append(apply_transformation(verts_cam_init, cam2world))
        init_joints.append(apply_transformation(joints_cam_init, cam2world))
        final_verts.append(apply_transformation(verts_cam, cam2world))
        final_joints.append(apply_transformation(joints_cam, cam2world))

        mosh_verts, mosh_joints = load_mosh_body(prox_root, recording, frame, gt_body_model, device)
        gt_verts.append(apply_transformation(mosh_verts, vicon2scene))
        gt_joints.append(apply_transformation(mosh_joints, vicon2scene))

        contact, vecs = contact_from_pointcloud(
            prediction['scene_points'], prediction['scene_normals'],
            verts_cam, device=device)
        assert np.array_equal(contact, prediction['contact_pred_reference']), (
            f'contact mask does not match the value recorded at release time: {frame_path}')

        scene = build_mesh_scene(scene_mesh) if no_scene_cache else scene_gt
        gt_contact, gt_vec = contact_from_mesh_scene(scene, gt_verts[-1])

        pred_contacts.append(contact)
        deco_contacts.append(prediction['contact_deco'])
        gt_contacts.append(gt_contact)
        visibility.append(prediction['visibility_mask'])
        pred_vecs.append(vecs)
        gt_vecs.append(gt_vec)

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

    # Contact is scored over the body vertices that are visible in the image, over the
    # occluded ones, and over all of them.
    contact = {}
    for split in ('visible', 'occluded', 'all'):
        if split == 'visible':
            select = visibility
        elif split == 'occluded':
            select = [~v for v in visibility]
        else:
            select = [np.ones_like(v) for v in visibility]

        def gather(per_frame):
            return np.concatenate([c[s] for c, s in zip(per_frame, select)])

        gt_split = gather(gt_contacts)
        contact[split] = {'ours': contact_scores(gt_split, gather(pred_contacts)),
                          'deco': contact_scores(gt_split, gather(deco_contacts))}

    # Vertex-to-scene vectors are predicted in the camera frame; rotate them into the
    # scene frame where the ground-truth vectors were measured.
    pred_vecs_scene = [v @ cam2world[:3, :3].T for v in pred_vecs]
    vectors = {
        'v2s_l2_m': weighted_contact_vector_l2(
            pred_vecs_scene, gt_vecs, contact_vertices_mask, contact_vertices_weights),
        'v2s_angle_deg': weighted_contact_vector_angle_deg(
            pred_vecs_scene, gt_vecs, contact_vertices_mask, contact_vertices_weights),
    }

    return {'dataset': 'PROX-quantitative', 'num_frames': len(frame_paths),
            'pose': pose, 'contact': contact, 'vectors': vectors}
