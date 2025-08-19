"""
Core inference and utility functions for CellSAM
"""

import logging

import numpy as np
import torch
from tqdm import trange

from .dynamics import compute_masks
from .transforms import convert_image, normalize_img, resize_image
from .utils import TqdmToLogger

logger = logging.getLogger(__name__)


def assign_device(gpu=True, device=None):
    """
    Assign computation device (CPU/GPU/MPS)

    Args:
        gpu (bool): Whether to use GPU if available
        device: Specific device to use

    Returns:
        torch.device: Selected device
    """
    if device is not None:
        return device

    if not gpu:
        logger.info("Using CPU")
        return torch.device("cpu")

    if torch.cuda.is_available():
        logger.info("Using GPU (CUDA)")
        return torch.device("cuda:0")

    if torch.backends.mps.is_available():
        logger.info("Using GPU (MPS)")
        return torch.device("mps")

    logger.info("GPU not available, using CPU")
    return torch.device("cpu")


def _to_device(x, device, dtype):
    """Move tensor to device with specified dtype"""
    X = torch.from_numpy(x).to(device)
    if dtype is not None:
        X = X.type(dtype)
    return X


def _from_device(x):
    """Move tensor from device to numpy"""
    # Convert BFloat16 to float32 before converting to numpy
    if x.dtype == torch.bfloat16:
        x = x.float()
    return x.detach().cpu().numpy()


def run_network(net, x):
    """
    Run network on input tensor

    Args:
        net: Neural network model
        x (numpy.ndarray): Input images

    Returns:
        tuple: (predictions, style_features)
    """
    X = _to_device(x, device=net.device, dtype=net.dtype)
    net.eval()
    with torch.no_grad():
        y, style = net(X)[:2]
    del X
    y = _from_device(y)
    style = _from_device(style)
    return y, style


def run_tiled_prediction(
    net, img, batch_size=8, tile_overlap=0.1, bsize=256, augment=False
):
    """
    Run network on image with tiling for memory efficiency using Cellpose-style processing

    Args:
        net: Neural network model
        img (np.ndarray): Input image [Lz, Ly, Lx, nchan]
        batch_size (int): Number of tiles to process simultaneously
        tile_overlap (float): Fraction of overlap between tiles
        bsize (int): Tile size in pixels
        augment (bool): Use test-time augmentation

    Returns:
        tuple: (predictions, style_features)
    """
    from .transforms import average_tiles, get_pad_yx, make_tiles, unaugment_tiles

    Lz, Ly0, Lx0, nchan = img.shape

    # Pad image to ensure it's divisible by bsize
    ypad1, ypad2, xpad1, xpad2 = get_pad_yx(Ly0, Lx0, min_size=(bsize, bsize))
    Ly, Lx = Ly0 + ypad1 + ypad2, Lx0 + xpad1 + xpad2

    # Create padded image
    img_padded = np.zeros((Lz, Ly, Lx, nchan), dtype=img.dtype)
    img_padded[:, ypad1 : ypad1 + Ly0, xpad1 : xpad1 + Lx0] = img

    # Initialize output arrays
    y_total = np.zeros((Lz, Ly, Lx, 3), dtype=np.float32)
    style_total = np.zeros((Lz, 256), dtype=np.float32)

    # Process each Z slice using Cellpose-style tiling
    for z in range(Lz):
        # Convert to channel-first format for make_tiles
        imgz = img_padded[z].transpose(2, 0, 1)  # HWC -> CHW

        # Create tiles using Cellpose's make_tiles function
        IMG, ysub, xsub, Ly_tile, Lx_tile = make_tiles(
            imgz, bsize=bsize, augment=augment, tile_overlap=tile_overlap
        )

        ntiles = IMG.shape[0]

        # Process tiles in batches
        ya = np.zeros((ntiles, 3, bsize, bsize), dtype=np.float32)
        stylea = np.zeros((ntiles, 256), dtype=np.float32)

        for j in range(0, ntiles, batch_size):
            bslc = slice(j, min(j + batch_size, ntiles))
            batch_tiles = IMG[bslc]

            # Run network
            ya_batch, stylea_batch = run_network(net, batch_tiles)
            ya[bslc] = ya_batch
            stylea[bslc] = stylea_batch

        # Handle augmentation reversal if needed
        if augment:
            # Calculate grid dimensions
            ny = (
                int(np.ceil(2.0 * Ly_tile / bsize))
                if augment
                else int(np.ceil((1.0 + 2 * tile_overlap) * Ly_tile / bsize))
            )
            nx = (
                int(np.ceil(2.0 * Lx_tile / bsize))
                if augment
                else int(np.ceil((1.0 + 2 * tile_overlap) * Lx_tile / bsize))
            )

            if ntiles == ny * nx:  # Only if we have the expected grid
                y_grid = np.reshape(ya, (ny, nx, 3, bsize, bsize))
                y_grid = unaugment_tiles(y_grid)
                ya = np.reshape(y_grid, (-1, 3, bsize, bsize))

        # Average tiles using Cellpose's proper blending
        yf = average_tiles(ya, ysub, xsub, Ly_tile, Lx_tile)
        y_total[z] = yf.transpose(1, 2, 0)  # CHW -> HWC

        # Average styles
        style_total[z] = stylea.mean(axis=0)

    # Remove padding
    y_final = y_total[:, ypad1 : ypad1 + Ly0, xpad1 : xpad1 + Lx0]

    return y_final, style_total


def run_inference(
    net,
    images,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    do_3D=False,
    anisotropy=None,
    batch_size=8,
    normalize=True,
    tile_overlap=0.1,
    augment=False,
    min_size=15,
    max_size_fraction=0.4,
):
    """
    Main inference function for CellSAM

    Args:
        net: CellSAM network
        images: Input images (numpy array or list)
        diameter (float): Expected cell diameter
        flow_threshold (float): Flow error threshold
        cellprob_threshold (float): Cell probability threshold
        do_3D (bool): Perform 3D segmentation
        anisotropy (float): Z-axis scaling factor
        batch_size (int): Batch size for processing
        normalize (bool): Normalize intensities
        tile_overlap (float): Tile overlap fraction
        augment (bool): Use test-time augmentation
        min_size (int): Minimum mask size
        max_size_fraction (float): Maximum mask size as fraction of image

    Returns:
        tuple: (masks, flows, styles)
    """
    # Handle list of images
    if isinstance(images, list):
        masks, flows, styles = [], [], []
        tqdm_out = TqdmToLogger(logger)
        for i in trange(len(images), file=tqdm_out, mininterval=30):
            mask, flow, style = run_inference(
                net,
                images[i],
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                do_3D=do_3D,
                anisotropy=anisotropy,
                batch_size=batch_size,
                normalize=normalize,
                tile_overlap=tile_overlap,
                augment=augment,
                min_size=min_size,
                max_size_fraction=max_size_fraction,
            )
            masks.append(mask)
            flows.append(flow)
            styles.append(style)
        return masks, flows, styles

    # Convert and normalize image
    x = convert_image(images, do_3D=do_3D)

    if normalize:
        x = normalize_img(x)

    # Resize image if diameter specified
    original_shape = x.shape
    if diameter is not None:
        scale_factor = 30.0 / diameter
        x = resize_image(x, scale_factor)

    # Run network prediction
    if x.size > 256 * 256 * 4:  # Use tiling for large images
        flows, styles = run_tiled_prediction(
            net, x, batch_size=batch_size, tile_overlap=tile_overlap, augment=augment
        )
    else:
        x_tensor = np.transpose(x, (0, 3, 1, 2))  # ZHWC -> ZCHW
        flows, styles = run_network(net, x_tensor)
        flows = np.transpose(flows, (0, 2, 3, 1))  # ZCHW -> ZHWC

    # Resize flows back to original size if needed
    if diameter is not None:
        flows = resize_image(flows, 1.0 / scale_factor)

    # Compute masks from flows
    masks = compute_masks(
        flows,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        do_3D=do_3D,
        min_size=min_size,
        max_size_fraction=max_size_fraction,
    )

    return masks, flows, styles
