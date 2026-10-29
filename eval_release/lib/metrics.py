"""Metric definitions: Procrustes-aligned joint error, and contact scores.

The Procrustes code is adapted from HMR (https://github.com/akanazawa/hmr) via
CameraHMR (https://github.com/pixelite1201/CameraHMR/), unchanged.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_similarity_transform(S1, S2):
    """Align S1 (3 x N) to S2 with a similarity transform (scale, R, t)."""
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1 ** 2)
    K = X1.dot(X2.T)

    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale * (R.dot(mu1))

    S1_hat = scale * R.dot(S1) + t
    if transposed:
        S1_hat = S1_hat.T
    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction='mean'):
    """Procrustes-aligned reconstruction error (PA-MPJPE) over a batch."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re_per_joint = np.sqrt(((S1_hat - S2) ** 2).sum(axis=-1))
    re = re_per_joint
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re, re_per_joint


def mean_point_error(pred, gt):
    """MPJPE / MPVPE in millimetres for stacked (B, N, 3) arrays in metres."""
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000)


def pa_mpjpe(pred, gt):
    """PA-MPJPE in millimetres for stacked (B, J, 3) arrays in metres."""
    return float(reconstruction_error(pred, gt)[0] * 1000)


def contact_scores(gt_contacts, pred_contacts):
    """Binary contact scores over a flat concatenation of all body vertices."""
    gt = np.asarray(gt_contacts).reshape(-1) > 0.5
    pred = np.asarray(pred_contacts).reshape(-1) > 0.5
    return {
        'accuracy': float(accuracy_score(gt, pred)),
        'precision': float(precision_score(gt, pred, zero_division=0)),
        'recall': float(recall_score(gt, pred, zero_division=0)),
        'f1': float(f1_score(gt, pred, zero_division=0)),
    }


def weighted_contact_vector_l2(pred_vecs_list, gt_vecs_list, mask, weights):
    """Weighted L2 error (metres) between predicted and GT vertex-to-scene vectors."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    per_frame = []
    for pred_vecs, gt_vecs in zip(pred_vecs_list, gt_vecs_list):
        pred = np.asarray(pred_vecs)[mask]
        gt = np.asarray(gt_vecs)[mask]
        l2_per_vertex = np.linalg.norm(pred - gt, axis=-1)
        per_frame.append(np.sum(l2_per_vertex * weights))
    return float(np.mean(per_frame))


def weighted_contact_vector_angle_deg(pred_vecs_list, gt_vecs_list, mask, weights, eps=1e-8):
    """Weighted angular error (degrees) between predicted and GT vertex-to-scene vectors."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / (w.sum() + 1e-12)

    per_frame = []
    for pred_vecs, gt_vecs in zip(pred_vecs_list, gt_vecs_list):
        pred = np.asarray(pred_vecs)[mask]
        gt = np.asarray(gt_vecs)[mask]

        denom = np.clip(np.linalg.norm(pred, axis=-1) * np.linalg.norm(gt, axis=-1), eps, None)
        cosang = np.clip(np.sum(pred * gt, axis=-1) / denom, -1.0, 1.0)
        per_frame.append(np.sum(np.degrees(np.arccos(cosang)) * w))

    return float(np.mean(per_frame))
