import torch
import numpy as np
from PIL import Image
from typing import List
from sklearn.linear_model import RANSACRegressor, LinearRegression


def clean():
    import gc

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def convert_numpy_to_pil(image_np):
    """
    Convert a numpy image to a PIL image.

    Args:
        image_np (np.array): a numpy image

    Returns:
        PIL image
    """
    image_pil = Image.fromarray(image_np)
    return image_pil


def mask_to_rectangle_mask(mask):
    """
    Convert a mask to a rectangle mask.

    Args:
        mask (np.array): a numpy array representing the mask

    Returns:
        np.array: a numpy array representing the rectangle mask
    """
    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    rect_mask = np.zeros_like(mask, dtype=np.uint8)
    rect_mask[y_min : y_max + 1, x_min : x_max + 1] = 255
    return rect_mask


def mask_to_square_mask(mask):
    """
    Convert a mask to a square mask (bbox of the mask).

    Args:
        mask (np.array): a numpy array representing the mask

    Returns:
        np.array: a numpy array representing the square mask
    """
    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    side = max(height, width)
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    half = side // 2
    y1, y2 = cy - half, cy - half + side
    x1, x2 = cx - half, cx - half + side
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = min(mask.shape[0], y2), min(mask.shape[1], x2)
    square_mask = np.zeros_like(mask, dtype=np.uint8)
    square_mask[y1:y2, x1:x2] = 255
    return square_mask


def mask_to_bbox(mask, margin=0.3):
    """
    Generate a bounding box around the non-zero regions of a binary mask with an additional margin.
    Parameters:
        mask (numpy.ndarray): A 2D array representing the binary mask, where non-zero values indicate the region of interest.
        margin (float, optional): The fraction of the detected bounding box dimensions to use as a margin around the object.
                                  Defaults to 0.3, meaning a 30% extension in both dimensions.
    Returns:
        tuple: A tuple of four integers (x_min, y_min, x_max, y_max) representing the bounding box coordinates.
               If the mask does not contain any non-zero values, the bounding box spans the entire mask.
    """

    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        bbox = (0, 0, mask.shape[1], mask.shape[0])
    else:
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        margin_x = int(margin * (max_x - min_x))
        margin_y = int(margin * (max_y - min_y))
        bbox = (
            max(0, min_x - margin_x),
            max(0, min_y - margin_y),
            min(mask.shape[1], max_x + margin_x),
            min(mask.shape[0], max_y + margin_y),
        )

    return bbox


def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2):
    print("Aligning depth maps using RANSAC...")
    regressor = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=True), min_samples=min_samples
    )
    if mask is not None:
        regressor.fit(
            relative_depth[mask].reshape(-1, 1), metric_depth[mask].reshape(-1, 1)
        )
    else:
        regressor.fit(relative_depth.reshape(-1, 1), metric_depth.reshape(-1, 1))

    print(
        "The estimated parameters are:",
        regressor.estimator_.coef_,
        regressor.estimator_.intercept_,
    )
    depth = regressor.predict(relative_depth.reshape(-1, 1)).reshape(
        relative_depth.shape
    )
    return depth


def align_depth_pt(
    relative_depth, metric_depth, mask=None, min_samples=0.2, iterations=20
):
    print("Aligning depth maps using PyTorch RANSAC...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert numpy arrays to torch tensors and move to device
    relative_depth = torch.from_numpy(relative_depth).float().to(device)
    metric_depth = torch.from_numpy(metric_depth).float().to(device)
    if mask is not None:
        mask = torch.from_numpy(mask).bool().to(device)

    # Select valid pixels
    if mask is not None:
        x = relative_depth[mask].reshape(-1, 1)
        y = metric_depth[mask].reshape(-1, 1)
    else:
        x = relative_depth.reshape(-1, 1)
        y = metric_depth.reshape(-1, 1)

    n = x.shape[0]
    # Determine sample size: if min_samples is fraction (<1), use fraction, else as count.
    s = int(min_samples * n) if min_samples < 1 else int(min_samples)
    s = max(s, 2)

    tol = torch.median(torch.abs(y - torch.median(y))).item()
    best_inliers = 0
    best_coef = None

    for _ in range(iterations):
        idx = torch.randperm(n)[:s]
        x_sample = x[idx]
        y_sample = y[idx]
        A = torch.cat((x_sample, torch.ones_like(x_sample)), dim=1)
        # Solve for [slope, intercept] using least squares
        coef = torch.linalg.lstsq(A, y_sample, rcond=None).solution
        residuals = torch.abs(x * coef[0] + coef[1] - y)
        inliers = (residuals < tol).sum().item()

        if inliers > best_inliers:
            best_inliers = inliers
            best_coef = coef

    # Recompute with all inliers based on best parameters
    y_pred = x * best_coef[0] + best_coef[1]
    inlier_mask = (torch.abs(y - y_pred) < tol).reshape(-1)
    if inlier_mask.sum() >= 2:
        A_final = torch.cat((x[inlier_mask], torch.ones_like(x[inlier_mask])), dim=1)
        best_coef = torch.linalg.lstsq(A_final, y[inlier_mask], rcond=None).solution

    print("The estimated parameters are:", best_coef[0].item(), best_coef[1].item())

    aligned = (relative_depth.reshape(-1, 1) * best_coef[0] + best_coef[1]).reshape(
        relative_depth.shape
    )
    return aligned.cpu().numpy()


def align_points_pt(
    affine_points,
    metric_points,
    mask=None,
    min_samples=0.2,
    iterations=20,
    device="cuda",
):
    """
    Align a set of 3D points (affine_points) to another set of 3D points (metric_points) using PyTorch and RANSAC.

    Args:
        affine_points (torch.Tensor or np.array): A tensor/array of shape (..., 3) representing points in affine space.
        metric_points (torch.Tensor or np.array): A tensor/array of shape (..., 3) representing points in metric space.
        mask (torch.Tensor or np.array, optional): A boolean array to filter valid points. Defaults to None.
        min_samples (float or int, optional): Minimum number or fraction of points for each RANSAC iteration. Defaults to 0.2.
        iterations (int, optional): Number of RANSAC iterations. Defaults to 20.
        device (str, optional): Device to run computations on. Defaults to 'cuda'.

    Returns:
        np.array: Transformed points aligned to the metric space.
    """

    if not torch.is_tensor(affine_points):
        affine_points = torch.as_tensor(affine_points, dtype=torch.float).to(device)
    if not torch.is_tensor(metric_points):
        metric_points = torch.as_tensor(metric_points, dtype=torch.float).to(device)
    if mask is not None and not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, dtype=torch.bool).to(device)

    device = affine_points.device
    valid = (
        torch.ones(affine_points.shape[:2], dtype=torch.bool, device=device)
        if mask is None
        else mask.bool()
    )
    pts_a = affine_points[valid].reshape(-1, 3)
    pts_m = metric_points[valid].reshape(-1, 3)
    n = pts_a.shape[0]
    s_count = int(min_samples * n) if min_samples < 1 else int(min_samples)
    s_count = max(s_count, 2)

    def solve_st(a, m):
        # Sums for closed-form solution
        A2 = torch.sum(a[:, 0] ** 2 + a[:, 1] ** 2 + a[:, 2] ** 2)
        MdotA = torch.sum(m[:, 0] * a[:, 0] + m[:, 1] * a[:, 1] + m[:, 2] * a[:, 2])
        Az = torch.sum(a[:, 2])
        Mz = torch.sum(m[:, 2])
        count = a.shape[0]
        num = MdotA - (Az * Mz / count)
        den = A2 - (Az**2 / count)
        scale = num / den
        shift = (Mz - scale * Az) / count
        return scale, shift

    best_inliers = 0
    best_scale = None
    best_shift = None

    # RANSAC
    for _ in range(iterations):
        inds = torch.randperm(n, device=device)[:s_count]
        s_temp, t_temp = solve_st(pts_a[inds], pts_m[inds])
        pred = pts_a * s_temp
        pred[:, 2] += t_temp
        dist = torch.norm(pred - pts_m, dim=1)
        median_dist = torch.median(dist)
        inliers = (dist < 1.5 * median_dist).sum().item()
        if inliers > best_inliers:
            best_inliers = inliers
            best_scale = s_temp
            best_shift = t_temp

    # Final solve
    if best_scale is not None:
        pred = pts_a * best_scale
        pred[:, 2] += best_shift
        inlier_mask = torch.norm(pred - pts_m, dim=1) < 1.5 * torch.median(
            torch.norm(pred - pts_m, dim=1)
        )
        if inlier_mask.sum() >= 2:
            best_scale, best_shift = solve_st(pts_a[inlier_mask], pts_m[inlier_mask])

    transformed = affine_points.clone()
    transformed[..., :2] *= best_scale
    transformed[..., 2] = transformed[..., 2] * best_scale + best_shift

    print(f"Scale: {best_scale}, Shift: {best_shift}")
    return transformed.cpu().numpy()


def focal_length_to_intrinsics(focal_length, image_size):
    """
    Convert focal length to camera intrinsics.

    Args:
        focal_length (float): focal length in pixels
        image_size (tuple): a tuple of (width, height)

    Returns:
        np.array: a numpy array representing the camera intrinsics
    """
    width, height = image_size
    K = np.array(
        [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]]
    )
    return K


def pack_points_into_batch(points: List[torch.Tensor]):
    """
    Packs a list of 3D point tensors into a single batch tensor with padding.
    Each tensor in the input list represents a set of 3D points with shape (num_points, 3). This function aggregates these
    tensors into a single batch tensor of shape (batch_size, max_num_points, 3), where max_num_points is the length of the
    longest tensor in the list. Padding with zeros is applied to tensors with fewer points. Additionally, a separate tensor is
    returned to track the original number of points in each batch element.
    Parameters:
        points (List[torch.Tensor]): List of tensors, each of shape (num_points, 3).
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - batch: A padded tensor containing all points, of shape (batch_size, max_num_points, 3).
            - lengths: A tensor containing the original number of points for each tensor in the input list.
    """

    batch_size = len(points)
    lengths = torch.tensor(
        [p.shape[0] for p in points], dtype=torch.long, device=points[0].device
    )
    max_len = lengths.max().item()
    batch = torch.zeros(
        (batch_size, max_len, 3), dtype=points[0].dtype, device=points[0].device
    )
    for i, p in enumerate(points):
        batch[i, : p.shape[0]] = p
    return batch, lengths


def find_closest_boxes_by_iou(
    original_boxes: np.ndarray, detected_boxes_res: np.ndarray
) -> np.ndarray:
    """
    For each box in original_boxes, find the closest box in detected_boxes_res based on IoU.

    Args:
        original_boxes (np.ndarray): Array of original boxes, shape (N, 4+), where the first 4 are (x1, y1, x2, y2).
        detected_boxes_res (np.ndarray): Array of detected boxes from NLF, shape (M, 4+).

    Returns:
        np.ndarray: An array of shape (N,) containing the indices of the closest box in
                    detected_boxes_res for each original box. -1 if no suitable match is found.
    """
    # If no boxes to match, return all -1
    N = original_boxes.shape[0]
    if N == 0 or detected_boxes_res.shape[0] == 0:
        return np.full((N,), -1, dtype=int)

    orig = original_boxes[:, :4]
    dets = detected_boxes_res[:, :4]
    matches = np.full((N,), -1, dtype=int)

    # Precompute areas
    area_orig = (orig[:, 2] - orig[:, 0]) * (orig[:, 3] - orig[:, 1])
    area_det = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])

    for i in range(N):
        print(f"Processing original box {i + 1}/{N}")
        x1 = np.maximum(orig[i, 0], dets[:, 0])
        y1 = np.maximum(orig[i, 1], dets[:, 1])
        x2 = np.minimum(orig[i, 2], dets[:, 2])
        y2 = np.minimum(orig[i, 3], dets[:, 3])

        inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
        inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
        inter_area = inter_w * inter_h
        print(f"Intersection area: {inter_area}")

        union_area = area_orig[i] + area_det - inter_area
        print(f"Union area: {union_area}")
        # avoid division by zero
        ious = np.where(union_area > 0, inter_area / union_area, 0.0)

        best_idx = np.argmax(ious)
        if ious[best_idx] > 0:
            matches[i] = best_idx

    return matches
