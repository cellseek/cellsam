"""
Image transformation utilities for CellSAM
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def _taper_mask(ly=224, lx=224, sig=7.5):
    """Create a taper mask for blending tile boundaries"""
    bsize = max(ly, lx)
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1 / (1 + np.exp((xm - (bsize / 2 - 20)) / sig))
    mask = mask[:ly, np.newaxis] * mask[np.newaxis, :lx]
    return mask


def average_tiles(y, ysub, xsub, Ly, Lx):
    """
    Average the results of the network over tiles using tapered masks.
    This is the key function from original Cellpose that handles smooth blending.

    Args:
        y (float): Output of cellpose network for each tile. Shape: [ntiles x nclasses x bsize x bsize]
        ysub (list): List of arrays with start and end of tiles in Y of length ntiles
        xsub (list): List of arrays with start and end of tiles in X of length ntiles
        Ly (int): Size of pre-tiled image in Y (may be larger than original image if image size is less than bsize)
        Lx (int): Size of pre-tiled image in X (may be larger than original image if image size is less than bsize)

    Returns:
        yf (float32): Network output averaged over tiles. Shape: [nclasses x Ly x Lx]
    """
    Navg = np.zeros((Ly, Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf


def make_tiles(imgi, bsize=256, augment=False, tile_overlap=0.1):
    """Make tiles of image to run at test-time.

    Args:
        imgi (np.ndarray): Array of shape (nchan, Ly, Lx) representing the input image.
        bsize (int, optional): Size of tiles. Defaults to 224.
        augment (bool, optional): Whether to flip tiles and set tile_overlap=2. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles. Defaults to 0.1.

    Returns:
        A tuple containing (IMG, ysub, xsub, Ly, Lx):
        IMG (np.ndarray): Array of shape (ntiles, nchan, bsize, bsize) representing the tiles.
        ysub (list): List of arrays with start and end of tiles in Y of length ntiles.
        xsub (list): List of arrays with start and end of tiles in X of length ntiles.
        Ly (int): Height of the input image.
        Lx (int): Width of the input image.
    """
    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize - Ly, Lx))), axis=1)
            Ly = bsize
        if Lx < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize - Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, :]
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = IMG[j, i, :, :, ::-1]
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]

    IMG = np.reshape(IMG, (-1, nchan, IMG.shape[-2], IMG.shape[-1]))
    return IMG, ysub, xsub, Ly, Lx


def unaugment_tiles(y):
    """Reverse test-time augmentations for averaging (includes flipping of flowsY and flowsX).

    Args:
        y (float32): Array of shape (ntiles_y, ntiles_x, chan, Ly, Lx) where chan = (flowsY, flowsX, cell prob).

    Returns:
        float32: Array of shape (ntiles_y, ntiles_x, chan, Ly, Lx).

    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j % 2 == 0 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, :]
                y[j, i, 0] *= -1
            elif j % 2 == 1 and i % 2 == 0:
                y[j, i] = y[j, i, :, :, ::-1]
                y[j, i, 1] *= -1
            elif j % 2 == 1 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, ::-1]
                y[j, i, 0] *= -1
                y[j, i, 1] *= -1
    return y


def get_pad_yx(Ly, Lx, min_size=(256, 256)):
    """Get padding amounts for image to reach minimum size"""
    min_ly, min_lx = min_size
    ypad1, ypad2 = 0, 0
    xpad1, xpad2 = 0, 0
    
    if Ly < min_ly:
        ypad = min_ly - Ly
        ypad1 = ypad // 2
        ypad2 = ypad - ypad1
    
    if Lx < min_lx:
        xpad = min_lx - Lx
        xpad1 = xpad // 2
        xpad2 = xpad - xpad1
        
    return ypad1, ypad2, xpad1, xpad2


def convert_image(image, channel_axis=None, z_axis=None, do_3D=False):
    """
    Convert image to standard format [Z, Y, X, C]

    Args:
        image (np.ndarray): Input image
        channel_axis (int): Axis containing channels
        z_axis (int): Axis containing Z dimension
        do_3D (bool): Whether this is 3D data

    Returns:
        np.ndarray: Converted image in [Z, Y, X, C] format
    """
    image = np.asarray(image)

    # Handle different input formats
    if image.ndim == 2:
        # 2D grayscale -> add Z and C dimensions
        image = image[np.newaxis, :, :, np.newaxis]
    elif image.ndim == 3:
        if do_3D:
            # 3D volume -> add C dimension
            image = image[:, :, :, np.newaxis]
        else:
            # 2D RGB -> add Z dimension and move C to end
            if channel_axis is None:
                # Assume channels are last if C <= 4
                if image.shape[-1] <= 4:
                    channel_axis = -1
                else:
                    channel_axis = 0

            if channel_axis == 0:
                image = np.transpose(image, (1, 2, 0))
            image = image[np.newaxis, :, :, :]
    elif image.ndim == 4:
        # Already 4D, ensure correct axis order
        if z_axis is None:
            z_axis = 0 if do_3D else None
        if channel_axis is None:
            channel_axis = -1

        # Transpose to [Z, Y, X, C] format
        axes = list(range(4))
        if channel_axis != -1 and channel_axis != 3:
            axes[3], axes[channel_axis] = axes[channel_axis], axes[3]
        if z_axis is not None and z_axis != 0:
            axes[0], axes[z_axis] = axes[z_axis], axes[0]
        image = np.transpose(image, axes)

    # Ensure we have 3 channels for SAM
    if image.shape[-1] == 1:
        # Grayscale -> RGB
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 2:
        # 2 channels -> add third channel as zeros
        zeros = np.zeros(image.shape[:-1] + (1,), dtype=image.dtype)
        image = np.concatenate([image, zeros], axis=-1)
    elif image.shape[-1] > 3:
        # More than 3 channels -> keep first 3
        image = image[..., :3]

    return image.astype(np.float32)


def normalize_img(
    image,
    lower_percentile=1,
    upper_percentile=99,
    tile_norm_blocksize=0,
    smooth_3d=True,
):
    """
    Normalize image intensities

    Args:
        image (np.ndarray): Input image
        lower_percentile (float): Lower percentile for normalization
        upper_percentile (float): Upper percentile for normalization
        tile_norm_blocksize (int): Block size for tile-based normalization
        smooth_3d (bool): Apply 3D smoothing for stacks

    Returns:
        np.ndarray: Normalized image
    """
    image = image.astype(np.float32)

    if tile_norm_blocksize > 0:
        # Tile-based normalization for large images
        return _tile_normalize(
            image, tile_norm_blocksize, lower_percentile, upper_percentile
        )

    # Global normalization
    for c in range(image.shape[-1]):
        channel = image[..., c]

        if smooth_3d and image.shape[0] > 1:
            # 3D smoothing for z-stacks
            channel = gaussian_filter(channel, sigma=0.5)

        # Calculate percentiles
        lower_val = np.percentile(channel, lower_percentile)
        upper_val = np.percentile(channel, upper_percentile)

        # Normalize
        if upper_val > lower_val:
            channel = (channel - lower_val) / (upper_val - lower_val)
            channel = np.clip(channel, 0, 1)

        image[..., c] = channel

    return image


def _tile_normalize(image, blocksize, lower_perc, upper_perc):
    """Apply tile-based normalization"""
    Lz, Ly, Lx, nc = image.shape

    # Calculate tile positions
    ny_blocks = int(np.ceil(Ly / blocksize))
    nx_blocks = int(np.ceil(Lx / blocksize))

    normalized = np.zeros_like(image)

    for c in range(nc):
        for i in range(ny_blocks):
            for j in range(nx_blocks):
                y1 = i * blocksize
                y2 = min((i + 1) * blocksize, Ly)
                x1 = j * blocksize
                x2 = min((j + 1) * blocksize, Lx)

                tile = image[:, y1:y2, x1:x2, c]

                lower_val = np.percentile(tile, lower_perc)
                upper_val = np.percentile(tile, upper_perc)

                if upper_val > lower_val:
                    tile = (tile - lower_val) / (upper_val - lower_val)
                    tile = np.clip(tile, 0, 1)

                normalized[:, y1:y2, x1:x2, c] = tile

    return normalized


def resize_image(image, scale_factor, interpolation=None):
    """
    Resize image by scale factor

    Args:
        image (np.ndarray): Input image [Z, Y, X, C]
        scale_factor (float or tuple): Scaling factor(s)
        interpolation: Interpolation method

    Returns:
        np.ndarray: Resized image
    """
    if isinstance(scale_factor, (int, float)):
        scale_factor = [scale_factor, scale_factor]

    Lz, Ly, Lx, nc = image.shape
    new_Ly = int(Ly * scale_factor[0])
    new_Lx = int(Lx * scale_factor[1])

    if interpolation is None:
        interpolation = cv2.INTER_LINEAR

    resized = np.zeros((Lz, new_Ly, new_Lx, nc), dtype=image.dtype)

    for z in range(Lz):
        for c in range(nc):
            resized[z, :, :, c] = cv2.resize(
                image[z, :, :, c], (new_Lx, new_Ly), interpolation=interpolation
            )

    return resized


def augment_tiles(tile, flip_axis=None):
    """
    Apply augmentation to image tile

    Args:
        tile (np.ndarray): Input tile
        flip_axis (int): Axis to flip for augmentation

    Returns:
        np.ndarray: Augmented tile
    """
    if flip_axis is not None:
        tile = np.flip(tile, axis=flip_axis)

    return tile


def crop_image(image, crop_size, center=None):
    """
    Crop image to specified size

    Args:
        image (np.ndarray): Input image
        crop_size (tuple): Target crop size (H, W)
        center (tuple): Center point for crop

    Returns:
        np.ndarray: Cropped image
    """
    Lz, Ly, Lx, nc = image.shape
    crop_h, crop_w = crop_size

    if center is None:
        # Center crop
        start_y = (Ly - crop_h) // 2
        start_x = (Lx - crop_w) // 2
    else:
        center_y, center_x = center
        start_y = max(0, min(Ly - crop_h, center_y - crop_h // 2))
        start_x = max(0, min(Lx - crop_w, center_x - crop_w // 2))

    end_y = start_y + crop_h
    end_x = start_x + crop_w

    return image[:, start_y:end_y, start_x:end_x, :]


def pad_image(image, pad_size, mode="constant", constant_values=0):
    """
    Pad image to specified size

    Args:
        image (np.ndarray): Input image [Z, Y, X, C]
        pad_size (tuple): Target size after padding
        mode (str): Padding mode
        constant_values: Value for constant padding

    Returns:
        np.ndarray: Padded image
    """
    Lz, Ly, Lx, nc = image.shape
    target_y, target_x = pad_size

    ypad = max(0, target_y - Ly)
    xpad = max(0, target_x - Lx)

    ypad1 = ypad // 2
    ypad2 = ypad - ypad1
    xpad1 = xpad // 2
    xpad2 = xpad - xpad1

    pad_width = ((0, 0), (ypad1, ypad2), (xpad1, xpad2), (0, 0))

    return np.pad(image, pad_width, mode=mode, constant_values=constant_values)
