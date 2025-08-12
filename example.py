#!/usr/bin/env python3
"""
Example script demonstrating how to use CellSAM for 2D image segmentation

This script loads an image called 'image.png', performs cellular segmentation
using CellSAM, and saves the results with visualization.
"""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import CellSAM components
from sam import CellSAM, load_image, save_masks
from sam.visualization import flow_to_rgb


def main():
    """Main function to demonstrate CellSAM usage"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define paths
    image_path = "image.jpg"
    output_dir = Path("segmentation_results")
    output_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Load the image
        logger.info(f"Loading image from {image_path}")
        image = load_image(image_path)
        logger.info(f"Loaded image with shape: {image.shape}")

        # Step 2: Initialize CellSAM model and perform segmentation
        logger.info("Initializing CellSAM model...")
        model = CellSAM()

        logger.info("Starting segmentation...")
        masks, flows, styles = model.segment(
            image,
            # Optional parameters for segmentation
            # diameter=30,  # Expected cell diameter in pixels
            flow_threshold=0.4,  # Flow error threshold
            cellprob_threshold=0.0,  # Cell probability threshold
        )

        logger.info(f"Segmentation complete. Found {len(np.unique(masks)) - 1} cells")

        # Handle mask shape - squeeze out extra dimensions if present
        if masks.ndim > 2:
            masks = np.squeeze(masks)

        # Step 3: Save the segmentation masks
        mask_path = output_dir / "segmentation_masks.tif"
        save_masks(masks, mask_path, save_flows=True, flows=flows)
        logger.info(f"Saved masks to {mask_path}")

        # Step 4: Create and save visualization
        logger.info("Creating visualization...")
        plt.figure(figsize=(15, 5))

        # Normalize image for display
        display_image = image.copy()
        if display_image.dtype != np.uint8:
            # Normalize to 0-1 range if not already uint8
            display_image = display_image.astype(np.float32)
            display_image = (display_image - display_image.min()) / (
                display_image.max() - display_image.min()
            )
        else:
            # Convert uint8 to float for consistent handling
            display_image = display_image.astype(np.float32) / 255.0

        # Plot 1: Original image
        plt.subplot(1, 3, 1)
        if len(display_image.shape) == 3 and display_image.shape[2] == 3:
            plt.imshow(display_image)
        else:
            # Handle grayscale or single channel images
            if len(display_image.shape) == 3:
                display_image = display_image[:, :, 0]  # Take first channel
            plt.imshow(display_image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        # Plot 2: Flow field visualization
        plt.subplot(1, 3, 2)
        if flows is not None:
            # Convert flows to RGB visualization
            flow_rgb = flow_to_rgb(flows)
            # Handle case where flow_rgb has extra dimensions
            if flow_rgb.ndim > 3:
                flow_rgb = np.squeeze(flow_rgb)
            plt.imshow(flow_rgb)
            plt.title("Flow Field")
        else:
            plt.imshow(display_image, cmap="gray")
            plt.title("Flow Field (N/A)")
        plt.axis("off")

        # Plot 3: Original image with red boundaries
        plt.subplot(1, 3, 3)

        # Create the overlay image
        if len(display_image.shape) == 2:
            overlay_image = np.stack([display_image] * 3, axis=-1)
        else:
            overlay_image = display_image.copy()

        # Ensure overlay_image is in the right format
        if overlay_image.max() <= 1:
            overlay_image = (overlay_image * 255).astype(np.uint8)
        else:
            overlay_image = overlay_image.astype(np.uint8)

        # Create boundaries for each cell
        for cell_id in np.unique(masks)[1:]:  # Skip background (0)
            # Create binary mask for this cell
            cell_mask = (masks == cell_id).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw red boundaries
            cv2.drawContours(
                overlay_image, contours, -1, (255, 0, 0), 2
            )  # Red color, thickness 2

        # Convert back to float for matplotlib
        overlay_image = overlay_image.astype(np.float32) / 255.0

        plt.imshow(overlay_image)
        plt.title(f"Overlay with Boundaries")
        plt.axis("off")

        plt.tight_layout()

        # Save the visualization
        viz_path = output_dir / "segmentation_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.show()
        logger.info(f"Saved visualization to {viz_path}")

        # Step 5: Print summary statistics
        print("\n" + "=" * 50)
        print("SEGMENTATION SUMMARY")
        print("=" * 50)
        print(f"Input image shape: {image.shape}")
        print(f"Number of cells detected: {len(np.unique(masks)) - 1}")
        print(f"Mask value range: {masks.min()} - {masks.max()}")

        # Calculate cell size statistics
        cell_sizes = []
        for cell_id in np.unique(masks)[1:]:  # Exclude background
            cell_size = np.sum(masks == cell_id)
            cell_sizes.append(cell_size)

        if cell_sizes:
            print(f"Average cell size: {np.mean(cell_sizes):.1f} pixels")
            print(f"Cell size range: {min(cell_sizes)} - {max(cell_sizes)} pixels")

        print(f"\nResults saved to: {output_dir.absolute()}")
        print("=" * 50)

    except FileNotFoundError:
        logger.error(f"Image file '{image_path}' not found!")
        logger.info("Please ensure 'image.png' exists in the current directory")
        print("\nTo use this script:")
        print("1. Place your image file named 'image.png' in the same directory")
        print("2. Run: python example.py")

    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
