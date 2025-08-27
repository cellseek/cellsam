"""
Dynamics and mask computation for CellSAM
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


def compute_masks(
    flows,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    do_3D=False,
    min_size=15,
    max_size_fraction=0.4,
    niter=None,
):
    """
    Compute segmentation masks from flow fields

    Args:
        flows (np.ndarray): Flow predictions [Z, Y, X, 3]
                           (last dim: dy, dx, cellprob)
        flow_threshold (float): Flow error threshold
        cellprob_threshold (float): Cell probability threshold
        do_3D (bool): Perform 3D segmentation
        min_size (int): Minimum mask size in pixels
        max_size_fraction (float): Maximum mask size as fraction of image
        niter (int): Number of iterations for dynamics

    Returns:
        np.ndarray: Segmentation masks
    """
    if do_3D:
        return _compute_masks_3d(
            flows,
            flow_threshold,
            cellprob_threshold,
            min_size,
            max_size_fraction,
            niter,
        )
    else:
        # Process each 2D slice
        masks = []
        for z in range(flows.shape[0]):
            mask = _compute_masks_2d(
                flows[z],
                flow_threshold,
                cellprob_threshold,
                min_size,
                max_size_fraction,
                niter,
            )
            masks.append(mask)

        # If only one slice, return 2D array instead of 3D
        if len(masks) == 1:
            return masks[0]
        else:
            return np.stack(masks, axis=0)


def _compute_masks_2d(
    flow,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    max_size_fraction=0.4,
    niter=None,
):
    """Compute 2D masks from single flow field"""

    # Extract flow components
    dy = flow[:, :, 0]
    dx = flow[:, :, 1]
    cellprob = flow[:, :, 2]

    # Apply cell probability threshold
    cell_mask = cellprob > cellprob_threshold

    if not cell_mask.any():
        return np.zeros(flow.shape[:2], dtype=np.int32)

    # Run dynamics to find cell centers
    if niter is None:
        niter = 250

    centers = _run_dynamics_2d(dy, dx, cell_mask, niter)

    # Filter centers by flow error
    valid_centers = _filter_centers_by_flow(centers, dy, dx, flow_threshold)

    if len(valid_centers) == 0:
        return np.zeros(flow.shape[:2], dtype=np.int32)

    # Create masks using watershed
    masks = _watershed_from_centers(valid_centers, cell_mask, dy, dx)

    # Filter masks by size
    masks = _filter_masks_by_size(masks, min_size, max_size_fraction)

    return masks


def _compute_masks_3d(
    flows,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    max_size_fraction=0.4,
    niter=None,
):
    """Compute 3D masks from flow fields"""

    # Extract components (assuming 3D flows have dz, dy, dx, cellprob)
    if flows.shape[-1] == 3:
        # 2D flows in 3D stack
        return _compute_masks_2d_stack(
            flows,
            flow_threshold,
            cellprob_threshold,
            min_size,
            max_size_fraction,
            niter,
        )

    # True 3D flows would be implemented here
    # For now, fall back to 2D processing
    logger.warning("True 3D flow processing not implemented, using 2D slice-by-slice")
    return _compute_masks_2d_stack(
        flows, flow_threshold, cellprob_threshold, min_size, max_size_fraction, niter
    )


def _compute_masks_2d_stack(
    flows, flow_threshold, cellprob_threshold, min_size, max_size_fraction, niter
):
    """Process 3D stack as series of 2D slices"""
    masks = []
    for z in range(flows.shape[0]):
        mask = _compute_masks_2d(
            flows[z],
            flow_threshold,
            cellprob_threshold,
            min_size,
            max_size_fraction,
            niter,
        )
        masks.append(mask)

    masks = np.stack(masks, axis=0)

    # Optional: link masks across slices for 3D consistency
    # This would require tracking cell IDs across slices

    return masks


def _run_dynamics_2d(dy, dx, mask, niter=200):
    """
    Run dynamics simulation to find cell centers

    Args:
        dy, dx (np.ndarray): Flow fields
        mask (np.ndarray): Cell probability mask
        niter (int): Number of iterations

    Returns:
        list: List of (y, x) center coordinates
    """
    h, w = dy.shape

    # Initialize grid of coordinates
    y, x = np.mgrid[0:h, 0:w]
    y = y.astype(np.float32)
    x = x.astype(np.float32)

    # Only track pixels in mask
    mask_coords = np.where(mask)
    if len(mask_coords[0]) == 0:
        return []

    y_track = y[mask_coords].copy()
    x_track = x[mask_coords].copy()

    # Run dynamics simulation
    for t in range(niter):
        # Get current flow values by interpolation
        dy_interp = _interpolate_flow(dy, y_track, x_track)
        dx_interp = _interpolate_flow(dx, y_track, x_track)

        # Update positions
        y_track += dy_interp * 0.5  # Step size
        x_track += dx_interp * 0.5

        # Keep within bounds
        y_track = np.clip(y_track, 0, h - 1)
        x_track = np.clip(x_track, 0, w - 1)

    # Find unique endpoints as centers
    centers = []
    y_final = np.round(y_track).astype(int)
    x_final = np.round(x_track).astype(int)

    unique_coords = set(zip(y_final, x_final))

    for y_c, x_c in unique_coords:
        # Count how many trajectories ended here
        count = np.sum((y_final == y_c) & (x_final == x_c))
        if count > 5:  # Minimum convergence threshold
            centers.append((y_c, x_c))

    return centers


def _interpolate_flow(flow, y_coords, x_coords):
    """Interpolate flow values at given coordinates"""
    h, w = flow.shape

    # Bilinear interpolation
    y_coords = np.clip(y_coords, 0, h - 1.001)
    x_coords = np.clip(x_coords, 0, w - 1.001)

    y0 = np.floor(y_coords).astype(int)
    y1 = y0 + 1
    x0 = np.floor(x_coords).astype(int)
    x1 = x0 + 1

    y1 = np.clip(y1, 0, h - 1)
    x1 = np.clip(x1, 0, w - 1)

    fy = y_coords - y0
    fx = x_coords - x0

    # Get flow values at corners
    f00 = flow[y0, x0]
    f01 = flow[y0, x1]
    f10 = flow[y1, x0]
    f11 = flow[y1, x1]

    # Bilinear interpolation
    f0 = f00 * (1 - fx) + f01 * fx
    f1 = f10 * (1 - fx) + f11 * fx

    return f0 * (1 - fy) + f1 * fy


def _filter_centers_by_flow(centers, dy, dx, threshold):
    """Filter centers based on flow error"""
    valid_centers = []

    for y, x in centers:
        # Calculate flow error in local neighborhood
        y1, y2 = max(0, y - 5), min(dy.shape[0], y + 6)
        x1, x2 = max(0, x - 5), min(dy.shape[1], x + 6)

        local_dy = dy[y1:y2, x1:x2].astype(np.float32)
        local_dx = dx[y1:y2, x1:x2].astype(np.float32)

        # Flow should point toward center
        yy, xx = np.mgrid[y1:y2, x1:x2]
        expected_dy = (y - yy).astype(np.float32)
        expected_dx = (x - xx).astype(np.float32)

        # Normalize
        norm = np.sqrt(expected_dy**2 + expected_dx**2) + 1e-6
        expected_dy /= norm
        expected_dx /= norm

        flow_norm = np.sqrt(local_dy**2 + local_dx**2) + 1e-6
        local_dy /= flow_norm
        local_dx /= flow_norm

        # Calculate error
        error = np.mean(
            np.sqrt((local_dy - expected_dy) ** 2 + (local_dx - expected_dx) ** 2)
        )

        if error < threshold:
            valid_centers.append((y, x))

    return valid_centers


def _watershed_from_centers(centers, mask, dy, dx):
    """Create masks using watershed from centers"""
    h, w = mask.shape

    # Create markers for watershed
    markers = np.zeros((h, w), dtype=np.int32)
    for i, (y, x) in enumerate(centers):
        markers[y, x] = i + 1

    # Create distance transform for watershed
    # Use negative cell probability as "elevation"
    elevation = -mask.astype(np.float32)

    # Apply Gaussian smoothing
    elevation = gaussian_filter(elevation, sigma=1.0)

    # Run watershed
    labels = watershed(elevation, markers, mask=mask)

    return labels


def _filter_masks_by_size(masks, min_size, max_size_fraction):
    """Filter masks by size constraints"""
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]

    total_pixels = masks.size
    max_size = int(total_pixels * max_size_fraction)

    filtered_masks = np.zeros_like(masks)
    new_label = 1

    for label in unique_labels:
        mask_size = np.sum(masks == label)

        if min_size <= mask_size <= max_size:
            filtered_masks[masks == label] = new_label
            new_label += 1

    return filtered_masks


def flow_to_rgb(dy, dx, cellprob=None):
    """
    Convert flow fields to RGB visualization

    Args:
        dy, dx (np.ndarray): Flow components
        cellprob (np.ndarray): Cell probability (optional)

    Returns:
        np.ndarray: RGB image
    """
    # Calculate flow magnitude and angle
    magnitude = np.sqrt(dy**2 + dx**2)
    angle = np.arctan2(dy, dx)

    # Convert to HSV
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    saturation = np.ones_like(magnitude)
    value = magnitude / (np.max(magnitude) + 1e-6)

    # Apply cell probability mask if provided
    if cellprob is not None:
        value *= cellprob / (np.max(cellprob) + 1e-6)

    # Convert HSV to RGB
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    return rgb


def compute_flow_error(dy_true, dx_true, dy_pred, dx_pred, masks):
    """
    Compute flow prediction error

    Args:
        dy_true, dx_true (np.ndarray): Ground truth flows
        dy_pred, dx_pred (np.ndarray): Predicted flows
        masks (np.ndarray): Ground truth masks

    Returns:
        float: Mean flow error
    """
    # Only compute error where masks exist
    mask_pixels = masks > 0

    if not mask_pixels.any():
        return float("inf")

    dy_err = dy_true[mask_pixels] - dy_pred[mask_pixels]
    dx_err = dx_true[mask_pixels] - dx_pred[mask_pixels]

    flow_error = np.sqrt(dy_err**2 + dx_err**2)

    return np.mean(flow_error)
