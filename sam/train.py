"""
Training functionality for CellSAM
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sam.model import CellSAMNet
from sam.transforms import normalize_img, resize_image
from sam.utils import setup_logger


class CellSAMDataset(Dataset):
    """
    Dataset for CellSAM training
    """

    def __init__(
        self, images, masks, flows=None, transform=None, image_size=512, channels=[0, 1]
    ):
        """
        Args:
            images (list): List of training images
            masks (list): List of ground truth masks
            flows (list): List of ground truth flows (optional)
            transform: Data augmentation transforms
            image_size (int): Target image size
            channels (list): Which channels to use
        """
        self.images = images
        self.masks = masks
        self.flows = flows
        self.transform = transform
        self.image_size = image_size
        self.channels = channels

        # Validate data
        assert len(images) == len(masks), "Number of images and masks must match"
        if flows is not None:
            assert len(images) == len(flows), "Number of images and flows must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        # Select channels
        if len(image.shape) == 3 and image.shape[-1] > len(self.channels):
            image = image[..., self.channels]

        # Resize if needed
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = resize_image(image, self.image_size, self.image_size)
            mask = resize_image(
                mask, self.image_size, self.image_size, interpolation="nearest"
            )

        # Normalize image
        image = normalize_img(image)

        # Generate flows if not provided
        if self.flows is not None:
            flows = self.flows[idx].copy()
            if flows.shape[0] != self.image_size or flows.shape[1] != self.image_size:
                # Resize flow components separately
                flow_y = resize_image(flows[..., 0], self.image_size, self.image_size)
                flow_x = resize_image(flows[..., 1], self.image_size, self.image_size)
                if flows.shape[-1] > 2:
                    cellprob = resize_image(
                        flows[..., 2], self.image_size, self.image_size
                    )
                    flows = np.stack([flow_y, flow_x, cellprob], axis=-1)
                else:
                    flows = np.stack([flow_y, flow_x], axis=-1)
        else:
            # Generate flows from masks (placeholder - would need proper implementation)
            flows = self._generate_flows_from_masks(mask)

        # Apply transforms
        if self.transform:
            sample = self.transform({"image": image, "mask": mask, "flows": flows})
            image = sample["image"]
            mask = sample["mask"]
            flows = sample["flows"]

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        flows = torch.from_numpy(flows).float()

        # Ensure correct dimensions
        if image.ndim == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        elif image.ndim == 3:
            image = image.permute(2, 0, 1)  # HWC -> CHW

        if flows.ndim == 3:
            flows = flows.permute(2, 0, 1)  # HWC -> CHW

        return {"image": image, "mask": mask, "flows": flows}

    def _generate_flows_from_masks(self, mask):
        """Generate flow fields from masks (simplified version)"""
        # This is a placeholder - would need proper flow computation
        h, w = mask.shape
        flows = np.zeros((h, w, 3), dtype=np.float32)

        # Generate simple flows (center-pointing)
        y, x = np.mgrid[0:h, 0:w]

        for label in np.unique(mask):
            if label == 0:
                continue

            cell_mask = mask == label
            if np.sum(cell_mask) == 0:
                continue

            # Find centroid
            cy, cx = np.mean(np.where(cell_mask), axis=1)

            # Create flows pointing to center
            dy = cy - y
            dx = cx - x

            # Normalize by distance
            dist = np.sqrt(dy**2 + dx**2)
            dist[dist == 0] = 1

            flows[cell_mask, 0] = dy[cell_mask] / dist[cell_mask]
            flows[cell_mask, 1] = dx[cell_mask] / dist[cell_mask]
            flows[cell_mask, 2] = 1.0  # Cell probability

        return flows


class CellSAMLoss(nn.Module):
    """
    Loss function for CellSAM training
    """

    def __init__(self, flow_weight=1.0, cellprob_weight=1.0):
        super().__init__()
        self.flow_weight = flow_weight
        self.cellprob_weight = cellprob_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with 'flows' and 'cellprob'
            targets: Dict with 'flows' and 'mask'
        """
        losses = {}
        total_loss = 0

        # Flow loss
        if "flows" in predictions and "flows" in targets:
            pred_flows = predictions["flows"][:, :2]  # Only dy, dx
            target_flows = targets["flows"][:, :2]

            flow_loss = self.mse_loss(pred_flows, target_flows)
            losses["flow_loss"] = flow_loss
            total_loss += self.flow_weight * flow_loss

        # Cell probability loss
        if "cellprob" in predictions:
            pred_cellprob = predictions["cellprob"]

            if "flows" in targets and targets["flows"].shape[1] > 2:
                # Use cellprob from flows
                target_cellprob = targets["flows"][:, 2:3]
            else:
                # Generate from mask
                target_cellprob = (targets["mask"] > 0).float().unsqueeze(1)

            cellprob_loss = self.bce_loss(pred_cellprob, target_cellprob)
            losses["cellprob_loss"] = cellprob_loss
            total_loss += self.cellprob_weight * cellprob_loss

        losses["total_loss"] = total_loss
        return losses


class CellSAMTrainer:
    """
    Trainer class for CellSAM
    """

    def __init__(
        self,
        model,
        device="cuda",
        learning_rate=1e-4,
        weight_decay=1e-5,
        log_dir="./logs",
    ):
        """
        Args:
            model: CellSAMNet model
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay
            log_dir: Directory for logs
        """
        self.model = model
        self.device = device
        self.model.to(device)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Setup loss function
        self.criterion = CellSAMLoss()

        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = setup_logger("CellSAMTrainer")

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.history = {
            "loss": [],
            "val_loss": [],
            "flow_loss": [],
            "cellprob_loss": [],
            "lr": [],
        }

    def train_epoch(self, train_loader, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            images = batch["image"].to(self.device)
            flows = batch["flows"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Calculate loss
            targets = {"flows": flows, "mask": masks}
            losses = self.criterion(outputs, targets)

            # Backward pass
            loss = losses["total_loss"]
            loss.backward()
            self.optimizer.step()

            if scheduler:
                scheduler.step()

            epoch_losses.append({k: v.item() for k, v in losses.items()})

            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Calculate average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([losses[key] for losses in epoch_losses])

        return avg_losses

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                images = batch["image"].to(self.device)
                flows = batch["flows"].to(self.device)
                masks = batch["mask"].to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                targets = {"flows": flows, "mask": masks}
                losses = self.criterion(outputs, targets)

                val_losses.append({k: v.item() for k, v in losses.items()})

        # Calculate average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([losses[key] for losses in val_losses])

        return avg_losses

    def train(
        self, train_loader, val_loader=None, epochs=100, save_freq=10, patience=20
    ):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_freq: Save model every N epochs
            patience: Early stopping patience
        """
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Training
            train_losses = self.train_epoch(train_loader, scheduler)
            epoch_time = time.time() - start_time

            # Validation
            val_losses = None
            if val_loader:
                val_losses = self.validate(val_loader)

            # Update history
            self.history["loss"].append(train_losses["total_loss"])
            self.history["flow_loss"].append(train_losses.get("flow_loss", 0))
            self.history["cellprob_loss"].append(train_losses.get("cellprob_loss", 0))
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if val_losses:
                self.history["val_loss"].append(val_losses["total_loss"])

            # Logging
            log_str = f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
            log_str += f'Loss: {train_losses["total_loss"]:.4f}'

            if val_losses:
                log_str += f', Val Loss: {val_losses["total_loss"]:.4f}'

            self.logger.info(log_str)

            # Save model
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"model_epoch_{epoch+1}.pth")

            # Early stopping
            current_loss = (
                val_losses["total_loss"] if val_losses else train_losses["total_loss"]
            )
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.logger.info("Training completed!")

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
        }

        save_path = self.log_dir / filename
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")


def create_data_loaders(
    train_images,
    train_masks,
    val_images=None,
    val_masks=None,
    batch_size=4,
    num_workers=4,
    image_size=512,
    channels=[0, 1],
):
    """
    Create data loaders for training

    Args:
        train_images: Training images
        train_masks: Training masks
        val_images: Validation images
        val_masks: Validation masks
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        channels: Image channels to use

    Returns:
        train_loader, val_loader (or None)
    """
    # Create datasets
    train_dataset = CellSAMDataset(
        train_images, train_masks, image_size=image_size, channels=channels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_images is not None and val_masks is not None:
        val_dataset = CellSAMDataset(
            val_images, val_masks, image_size=image_size, channels=channels
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def train_model(
    train_images,
    train_masks,
    val_images=None,
    val_masks=None,
    device="cuda",
    epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    save_dir="./models",
):
    """
    High-level training function

    Args:
        train_images: Training images
        train_masks: Training masks
        val_images: Validation images
        val_masks: Validation masks
        device: Training device
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Directory to save models

    Returns:
        Trained model
    """
    # Setup
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    logger = setup_logger("train_model")
    logger.info(f"Starting training with {len(train_images)} images")

    # Create model
    model = CellSAMNet()

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_images, train_masks, val_images, val_masks, batch_size=batch_size
    )

    # Create trainer
    trainer = CellSAMTrainer(
        model, device=device, learning_rate=learning_rate, log_dir=save_dir / "logs"
    )

    # Train
    trainer.train(
        train_loader, val_loader, epochs=epochs, save_freq=max(1, epochs // 10)
    )

    # Save final model
    final_path = save_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved: {final_path}")

    return model, trainer.history
