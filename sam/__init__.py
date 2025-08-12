"""
CellSAM: Standalone implementation of Cellpose-SAM for cellular segmentation
"""

from .model import CellSAM
from .utils import load_image, save_masks
from .visualization import plot_segmentation

__version__ = "1.0.0"
__author__ = "CellSAM Team"

__all__ = [
    "CellSAM",
    "load_image",
    "save_masks",
    "plot_segmentation",
]
