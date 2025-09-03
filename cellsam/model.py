"""
Core model implementation for CellSAM
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from .core import assign_device, run_inference

logger = logging.getLogger(__name__)


class CellSAMNet(nn.Module):
    """
    CellSAM neural network based on Segment Anything Model (SAM) architecture
    Modified for cellular segmentation with flow field prediction
    """

    def __init__(
        self,
        backbone="vit_l",
        patch_size=8,
        nout=3,
        block_size=256,
        dropout_rate=0.4,
        dtype=torch.float32,
    ):
        """
        Initialize CellSAM network

        Args:
            backbone (str): SAM backbone architecture ('vit_b', 'vit_l', 'vit_h')
            patch_size (int): Patch size for tokenization
            nout (int): Number of output channels (3 for flows + cell probability)
            block_size (int): Block size for position embeddings
            dropout_rate (float): Dropout rate for training
            dtype (torch.dtype): Data type for model weights
        """
        super(CellSAMNet, self).__init__()

        # Initialize SAM encoder
        self.encoder = sam_model_registry[backbone](checkpoint=None).image_encoder

        # Modify patch embedding for custom patch size
        w = self.encoder.patch_embed.proj.weight.detach()
        nchan = w.shape[0]
        self.patch_size = patch_size
        self.encoder.patch_embed.proj = nn.Conv2d(
            3, nchan, stride=patch_size, kernel_size=patch_size
        )
        self.encoder.patch_embed.proj.weight.data = w[
            :, :, :: 16 // patch_size, :: 16 // patch_size
        ]

        # Adjust position embeddings
        ds = (1024 // 16) // (block_size // patch_size)
        self.encoder.pos_embed = nn.Parameter(
            self.encoder.pos_embed[:, ::ds, ::ds], requires_grad=True
        )

        # Output head for flow prediction
        self.nout = nout
        self.out = nn.Conv2d(256, self.nout * patch_size**2, kernel_size=1)

        # Learnable upsampling weights
        self.W2 = nn.Parameter(
            torch.eye(self.nout * patch_size**2).reshape(
                self.nout * patch_size**2, self.nout, patch_size, patch_size
            ),
            requires_grad=False,
        )

        # Model parameters
        self.dropout_rate = dropout_rate
        self.diam_labels = nn.Parameter(torch.tensor([30.0]), requires_grad=False)
        self.diam_mean = nn.Parameter(torch.tensor([30.0]), requires_grad=False)
        self.dtype = dtype

        # Set global attention in all transformer blocks
        for blk in self.encoder.blocks:
            blk.window_size = 0

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input images [B, C, H, W]

        Returns:
            tuple: (flows, style) where flows are [B, 3, H, W] and style is [B, 256]
        """
        # Patch embedding
        x = self.encoder.patch_embed(x)

        # Add position embeddings
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        # Apply dropout during training
        if self.training and self.dropout_rate > 0:
            nlay = len(self.encoder.blocks)
            rdrop = (
                torch.rand((len(x), nlay), device=x.device)
                < torch.linspace(0, self.dropout_rate, nlay, device=x.device)
            ).to(x.dtype)
            for i, blk in enumerate(self.encoder.blocks):
                mask = rdrop[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = x * mask + blk(x) * (1 - mask)
        else:
            for blk in self.encoder.blocks:
                x = blk(x)

        # Apply neck
        x = self.encoder.neck(x.permute(0, 3, 1, 2))

        # Output head
        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride=self.patch_size, padding=0)

        # Return flows and dummy style vector for compatibility
        style = torch.randn((x.shape[0], 256), device=x.device)
        return x1, style

    def load_model(self, path, device, strict=False):
        """Load model weights from file"""
        state_dict = torch.load(path, map_location=device, weights_only=True)

        # Handle DataParallel/DistributedDataParallel weights
        keys = list(state_dict.keys())
        if keys and keys[0].startswith("module."):
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict=strict)
        else:
            self.load_state_dict(state_dict, strict=strict)

        if self.dtype != torch.float32:
            self.to(self.dtype)

    @property
    def device(self):
        """Get the device of the model"""
        return next(self.parameters()).device

    def save_model(self, filename):
        """Save model weights to file"""
        torch.save(self.state_dict(), filename)


class CellSAM:
    """
    Main CellSAM class for cellular segmentation

    Simple interface for segmenting cellular images using SAM-based architecture
    """

    def __init__(
        self,
        gpu=True,
        device=None,
        use_bfloat16=True,
        model_path="weights/cpsam",
    ):
        """
        Initialize CellSAM model

        Args:
            gpu (bool): Use GPU if available
            device (torch.device): Specific device to use
            use_bfloat16 (bool): Use 16-bit precision for faster inference
        """
        # Setup device
        self.device = assign_device(gpu=gpu, device=device)

        # Initialize network
        import torch

        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.net = CellSAMNet(dtype=dtype).to(self.device)

        # Load pretrained model
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.net.load_model(model_path, device=self.device)
        else:
            # logger.info(f"Downloading model cpsam")
            # download_model("cpsam", model_path)
            # self.net.load_model(model_path, device=self.device)
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")

    def segment(
        self,
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
    ):
        """
        Segment cellular images

        Args:
            images: Input images (numpy array or list of arrays)
            diameter (float): Expected cell diameter in pixels
            flow_threshold (float): Flow error threshold for mask generation
            cellprob_threshold (float): Cell probability threshold
            do_3D (bool): Perform 3D segmentation
            anisotropy (float): Z-axis scaling factor for 3D
            batch_size (int): Batch size for processing
            normalize (bool): Normalize image intensities
            tile_overlap (float): Overlap fraction for tiling
            augment (bool): Use test-time augmentation

        Returns:
            tuple: (masks, flows, styles) where masks are segmentation masks,
                   flows are flow fields, and styles are feature vectors
        """
        return run_inference(
            self.net,
            images,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=do_3D,
            anisotropy=anisotropy,
            batch_size=batch_size,
            normalize=normalize,
            tile_overlap=tile_overlap,
            augment=augment,
        )

    def train(self, train_data, test_data=None, **kwargs):
        """
        Train the model on custom data

        Args:
            train_data: Training data (images and masks)
            test_data: Optional test data for validation
            **kwargs: Additional training parameters
        """
        # Import training utilities when needed
        from cellsam.train import train_model

        return train_model(self.net, train_data, test_data, **kwargs)
