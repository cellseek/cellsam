"""
Utility functions for CellSAM
"""

import logging
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TqdmToLogger:
    """
    Output stream for TQDM which will output to logger module instead of stdout
    """

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)


def setup_logger(name=None, level=logging.INFO):
    """
    Setup logger for CellSAM

    Args:
        name (str): Logger name
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    if name is None:
        name = __name__

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def download_model(model_name, save_path, base_url=None):
    """
    Download pretrained model weights

    Args:
        model_name (str): Name of model to download
        save_path (str or Path): Where to save the model
        base_url (str): Base URL for model downloads
    """
    if base_url is None:
        base_url = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/"

    url = base_url + model_name
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Downloading {model_name} from {url}")

    # Download with progress bar
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)

    logger.info(f"Model saved to {save_path}")


def load_image(image_path, channels=None):
    """
    Load image from file

    Args:
        image_path (str or Path): Path to image file
        channels (list): Which channels to load (None for all)

    Returns:
        np.ndarray: Loaded image
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Try different loading methods based on file extension
    ext = image_path.suffix.lower()

    if ext in [".tif", ".tiff"]:
        # Load TIFF (potentially multi-channel/multi-page)
        return _load_tiff(image_path, channels)
    elif ext in [".npy", ".npz"]:
        # Load numpy array
        return np.load(image_path)
    else:
        # Load standard image formats
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            # Try PIL as fallback
            img = np.array(Image.open(image_path))

        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)


def _load_tiff(image_path, channels=None):
    """Load TIFF file (potentially multi-page)"""
    try:
        import tifffile

        img = tifffile.imread(str(image_path))
    except ImportError:
        # Fallback to PIL
        from PIL import Image

        img = Image.open(image_path)

        # Handle multi-page TIFF
        images = []
        try:
            while True:
                images.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if len(images) == 1:
            img = images[0]
        else:
            img = np.stack(images, axis=0)

    img = img.astype(np.float32)

    # Select specific channels if requested
    if channels is not None and len(img.shape) > 2:
        if img.shape[-1] > max(channels):
            img = img[..., channels]

    return img


def save_masks(masks, save_path, save_flows=False, flows=None):
    """
    Save segmentation masks to file

    Args:
        masks (np.ndarray): Segmentation masks
        save_path (str or Path): Output file path
        save_flows (bool): Whether to save flow fields
        flows (np.ndarray): Flow fields to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure masks are 2D for single-slice case
    if masks.ndim > 2:
        masks = np.squeeze(masks)

    # Save masks
    if save_path.suffix.lower() in [".npy"]:
        np.save(save_path, masks)
    elif save_path.suffix.lower() in [".tif", ".tiff"]:
        try:
            import tifffile

            tifffile.imwrite(str(save_path), masks.astype(np.uint16))
        except ImportError:
            # Fallback to PIL for single slice
            if masks.ndim == 2:
                Image.fromarray(masks.astype(np.uint16)).save(save_path)
            else:
                raise ImportError("tifffile required for multi-page TIFF")
    else:
        # Save as PNG/JPEG (only for 2D)
        if masks.ndim == 2:
            cv2.imwrite(str(save_path), masks.astype(np.uint16))
        else:
            raise ValueError("Multi-dimensional masks require .npy or .tif format")

    # Save flows if requested
    if save_flows and flows is not None:
        flow_path = save_path.parent / (save_path.stem + "_flows" + save_path.suffix)
        try:
            if save_path.suffix.lower() in [".npy"]:
                np.save(flow_path, flows)
            else:
                # Convert flows to uint8 for visualization
                flow_rgb = _flows_to_rgb(flows)
                # Ensure flow_rgb is properly formatted for OpenCV
                if flow_rgb.ndim == 3 and flow_rgb.shape[-1] == 3:
                    # Convert to uint8 and ensure proper data type
                    flow_rgb = np.clip(flow_rgb, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(flow_path), flow_rgb)
                else:
                    # Fallback to numpy save if conversion fails
                    np.save(flow_path.with_suffix(".npy"), flows)
        except Exception as e:
            logger.warning(f"Could not save flows: {e}. Saving as .npy instead.")
            np.save(flow_path.with_suffix(".npy"), flows)


def _flows_to_rgb(flows):
    """Convert flow fields to RGB visualization"""
    from .dynamics import flow_to_rgb

    # Handle case where flows have extra dimensions
    if flows.ndim > 3:
        flows = np.squeeze(flows)

    if flows.ndim == 4:
        # Process each slice
        rgb_flows = []
        for z in range(flows.shape[0]):
            dy, dx = flows[z, :, :, 0], flows[z, :, :, 1]
            cellprob = flows[z, :, :, 2] if flows.shape[-1] > 2 else None
            rgb = flow_to_rgb(dy, dx, cellprob)
            rgb_flows.append(rgb)
        return np.stack(rgb_flows, axis=0)
    else:
        dy, dx = flows[:, :, 0], flows[:, :, 1]
        cellprob = flows[:, :, 2] if flows.shape[-1] > 2 else None
        return flow_to_rgb(dy, dx, cellprob)


def normalize_image(image, lower_percentile=1, upper_percentile=99):
    """
    Simple image normalization

    Args:
        image (np.ndarray): Input image
        lower_percentile (float): Lower percentile for normalization
        upper_percentile (float): Upper percentile for normalization

    Returns:
        np.ndarray: Normalized image
    """
    image = image.astype(np.float32)

    # Calculate percentiles
    lower_val = np.percentile(image, lower_percentile)
    upper_val = np.percentile(image, upper_percentile)

    # Normalize
    if upper_val > lower_val:
        image = (image - lower_val) / (upper_val - lower_val)
        image = np.clip(image, 0, 1)

    return image


def get_image_info(image):
    """
    Get information about image dimensions and properties

    Args:
        image (np.ndarray): Input image

    Returns:
        dict: Image information
    """
    info = {
        "shape": image.shape,
        "dtype": image.dtype,
        "ndim": image.ndim,
        "size": image.size,
        "min": np.min(image),
        "max": np.max(image),
        "mean": np.mean(image),
        "std": np.std(image),
    }

    # Add dimension labels
    if image.ndim == 2:
        info["dimensions"] = "HW"
    elif image.ndim == 3:
        info["dimensions"] = "HWC or ZHW"
    elif image.ndim == 4:
        info["dimensions"] = "ZHWC"

    return info


def create_color_map(n_colors):
    """
    Create colormap for mask visualization

    Args:
        n_colors (int): Number of colors needed

    Returns:
        np.ndarray: Color map array
    """
    if n_colors <= 256:
        # Use matplotlib colormap
        try:
            import matplotlib.cm as cm

            cmap = cm.get_cmap("tab20", n_colors)
            colors = cmap(np.linspace(0, 1, n_colors))[:, :3]  # Remove alpha
            return (colors * 255).astype(np.uint8)
        except ImportError:
            pass

    # Fallback: generate random colors
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, (n_colors, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    return colors


def masks_to_colored(masks, colors=None):
    """
    Convert integer masks to colored visualization

    Args:
        masks (np.ndarray): Integer masks
        colors (np.ndarray): Color map

    Returns:
        np.ndarray: Colored mask visualization
    """
    if masks.ndim == 3:
        # Process each slice
        colored_masks = []
        for z in range(masks.shape[0]):
            colored = masks_to_colored(masks[z], colors)
            colored_masks.append(colored)
        return np.stack(colored_masks, axis=0)

    unique_labels = np.unique(masks)
    n_labels = len(unique_labels)

    if colors is None:
        colors = create_color_map(n_labels)

    colored = np.zeros(masks.shape + (3,), dtype=np.uint8)

    for i, label in enumerate(unique_labels):
        if i < len(colors):
            colored[masks == label] = colors[i]

    return colored


def memory_usage():
    """
    Get current memory usage information

    Returns:
        dict: Memory usage statistics
    """

    import psutil

    # Python memory
    process = psutil.Process()
    python_memory = process.memory_info().rss / 1024 / 1024  # MB

    # GPU memory if available
    gpu_memory = {}
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
                cached = torch.cuda.memory_reserved(i) / 1024 / 1024  # MB
                gpu_memory[f"cuda:{i}"] = {"allocated": allocated, "cached": cached}
    except ImportError:
        pass

    return {"python_memory_mb": python_memory, "gpu_memory": gpu_memory}
