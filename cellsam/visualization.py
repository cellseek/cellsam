"""
Visualization functions for CellSAM
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_segmentation(
    image,
    masks=None,
    flows=None,
    title="CellSAM Segmentation",
    figsize=(12, 8),
    show_flows=True,
    alpha=0.7,
):
    """
    Plot image with segmentation masks and flow fields

    Args:
        image (np.ndarray): Input image
        masks (np.ndarray): Segmentation masks
        flows (np.ndarray): Flow fields
        title (str): Plot title
        figsize (tuple): Figure size
        show_flows (bool): Whether to show flow fields
        alpha (float): Mask transparency
    """
    # Handle 3D images (show middle slice)
    if image.ndim == 3 and image.shape[-1] not in [3, 4]:
        z_mid = image.shape[0] // 2
        image = image[z_mid]
        if masks is not None:
            masks = masks[z_mid]
        if flows is not None:
            flows = flows[z_mid]

    # Setup subplots
    n_plots = 1
    if masks is not None:
        n_plots += 1
    if flows is not None and show_flows:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot original image
    axes[plot_idx].imshow(image, cmap="gray" if image.ndim == 2 else None)
    axes[plot_idx].set_title("Original Image")
    axes[plot_idx].axis("off")
    plot_idx += 1

    # Plot masks
    if masks is not None:
        overlay = plot_masks_on_image(image, masks, alpha=alpha)
        axes[plot_idx].imshow(overlay)
        axes[plot_idx].set_title(f"Segmentation ({np.max(masks)} cells)")
        axes[plot_idx].axis("off")
        plot_idx += 1

    # Plot flows
    if flows is not None and show_flows:
        flow_img = flow_to_rgb(flows)
        axes[plot_idx].imshow(flow_img)
        axes[plot_idx].set_title("Flow Field")
        axes[plot_idx].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_masks_on_image(image, masks, alpha=0.7, colors=None):
    """
    Overlay masks on image

    Args:
        image (np.ndarray): Background image
        masks (np.ndarray): Segmentation masks
        alpha (float): Overlay transparency
        colors (np.ndarray): Color map for masks

    Returns:
        np.ndarray: Image with overlaid masks
    """
    from .utils import masks_to_colored

    # Normalize image
    if image.ndim == 2:
        img_rgb = np.stack([image] * 3, axis=-1)
    else:
        img_rgb = image.copy()

    if img_rgb.max() > 1:
        img_rgb = img_rgb / img_rgb.max()

    # Create colored masks
    colored_masks = masks_to_colored(masks, colors)
    colored_masks = colored_masks.astype(np.float32) / 255.0

    # Create overlay
    mask_overlay = (masks > 0).astype(np.float32)
    overlay = (
        img_rgb * (1 - alpha * mask_overlay[..., None])
        + colored_masks * alpha * mask_overlay[..., None]
    )

    return np.clip(overlay, 0, 1)


def flow_to_rgb(flows, cellprob=None, max_flow=None):
    """
    Convert flow field to RGB visualization

    Args:
        flows (np.ndarray): Flow field (H, W, 2) or separate dy, dx
        cellprob (np.ndarray): Cell probability map
        max_flow (float): Maximum flow magnitude for normalization

    Returns:
        np.ndarray: RGB flow visualization
    """
    if isinstance(flows, tuple) or (isinstance(flows, np.ndarray) and flows.ndim == 1):
        dy, dx = flows[0], flows[1]
    else:
        dy, dx = flows[..., 0], flows[..., 1]

    # Calculate flow magnitude and angle
    magnitude = np.sqrt(dy**2 + dx**2)
    angle = np.arctan2(dy, dx)

    # Normalize
    if max_flow is None:
        max_flow = np.percentile(magnitude, 99)

    if max_flow > 0:
        magnitude = np.clip(magnitude / max_flow, 0, 1)

    # Convert to HSV
    hsv = np.zeros(dy.shape + (3,), dtype=np.float32)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # Hue: angle
    hsv[..., 1] = magnitude  # Saturation: magnitude
    hsv[..., 2] = 1.0  # Value: full brightness

    # Apply cell probability mask if available
    if cellprob is not None:
        hsv[..., 2] = cellprob

    # Convert HSV to RGB
    rgb = hsv_to_rgb(hsv)

    return (rgb * 255).astype(np.uint8)


def hsv_to_rgb(hsv):
    """Convert HSV to RGB"""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb = np.zeros_like(hsv)

    # Red dominant
    idx = (i % 6) == 0
    rgb[idx, 0] = v[idx]
    rgb[idx, 1] = t[idx]
    rgb[idx, 2] = p[idx]

    # Yellow dominant
    idx = (i % 6) == 1
    rgb[idx, 0] = q[idx]
    rgb[idx, 1] = v[idx]
    rgb[idx, 2] = p[idx]

    # Green dominant
    idx = (i % 6) == 2
    rgb[idx, 0] = p[idx]
    rgb[idx, 1] = v[idx]
    rgb[idx, 2] = t[idx]

    # Cyan dominant
    idx = (i % 6) == 3
    rgb[idx, 0] = p[idx]
    rgb[idx, 1] = q[idx]
    rgb[idx, 2] = v[idx]

    # Blue dominant
    idx = (i % 6) == 4
    rgb[idx, 0] = t[idx]
    rgb[idx, 1] = p[idx]
    rgb[idx, 2] = v[idx]

    # Magenta dominant
    idx = (i % 6) == 5
    rgb[idx, 0] = v[idx]
    rgb[idx, 1] = p[idx]
    rgb[idx, 2] = q[idx]

    return rgb


def plot_3d_segmentation(image, masks, z_slice=None, max_slices=16):
    """
    Plot 3D segmentation as grid of slices

    Args:
        image (np.ndarray): 3D image (Z, H, W)
        masks (np.ndarray): 3D masks (Z, H, W)
        z_slice (int): Specific slice to highlight
        max_slices (int): Maximum number of slices to show
    """
    n_slices = image.shape[0]

    if n_slices > max_slices:
        # Sample evenly spaced slices
        slice_indices = np.linspace(0, n_slices - 1, max_slices, dtype=int)
    else:
        slice_indices = np.arange(n_slices)

    n_show = len(slice_indices)
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for i, z_idx in enumerate(slice_indices):
        row = i // cols
        col = i % cols

        # Create overlay
        overlay = plot_masks_on_image(image[z_idx], masks[z_idx])

        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f"Z={z_idx}")
        axes[row, col].axis("off")

        # Highlight specific slice
        if z_slice is not None and z_idx == z_slice:
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    # Hide empty subplots
    for i in range(n_show, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")

    plt.suptitle(f"3D Segmentation ({n_slices} slices, {np.max(masks)} cells)")
    plt.tight_layout()
    plt.show()


def plot_training_metrics(metrics_history, save_path=None):
    """
    Plot training metrics

    Args:
        metrics_history (dict): Dictionary with training metrics
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    if "loss" in metrics_history:
        axes[0, 0].plot(metrics_history["loss"], label="Training Loss")
        if "val_loss" in metrics_history:
            axes[0, 0].plot(metrics_history["val_loss"], label="Validation Loss")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # Accuracy/IoU
    if "iou" in metrics_history:
        axes[0, 1].plot(metrics_history["iou"], label="IoU")
        if "val_iou" in metrics_history:
            axes[0, 1].plot(metrics_history["val_iou"], label="Validation IoU")
        axes[0, 1].set_title("IoU Score")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("IoU")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # Learning rate
    if "lr" in metrics_history:
        axes[1, 0].plot(metrics_history["lr"])
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].grid(True)

    # Other metrics
    if "flow_loss" in metrics_history:
        axes[1, 1].plot(metrics_history["flow_loss"], label="Flow Loss")
        if "cellprob_loss" in metrics_history:
            axes[1, 1].plot(metrics_history["cellprob_loss"], label="Cell Prob Loss")
        axes[1, 1].set_title("Component Losses")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_prediction_examples(
    images, true_masks, pred_masks, flows=None, n_examples=4, figsize=(16, 12)
):
    """
    Plot comparison of predictions vs ground truth

    Args:
        images (list): List of input images
        true_masks (list): List of ground truth masks
        pred_masks (list): List of predicted masks
        flows (list): List of predicted flows
        n_examples (int): Number of examples to show
        figsize (tuple): Figure size
    """
    n_examples = min(n_examples, len(images))
    n_cols = 4 if flows is not None else 3

    fig, axes = plt.subplots(n_examples, n_cols, figsize=figsize)
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_examples):
        img = images[i]
        true_mask = true_masks[i]
        pred_mask = pred_masks[i]

        # Handle 3D by taking middle slice
        if img.ndim == 3 and img.shape[-1] not in [3, 4]:
            z_mid = img.shape[0] // 2
            img = img[z_mid]
            true_mask = true_mask[z_mid]
            pred_mask = pred_mask[z_mid]
            if flows is not None:
                flow = flows[i][z_mid]
        else:
            flow = flows[i] if flows is not None else None

        # Original image
        axes[i, 0].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Ground truth
        true_overlay = plot_masks_on_image(img, true_mask)
        axes[i, 1].imshow(true_overlay)
        axes[i, 1].set_title(f"Ground Truth ({np.max(true_mask)} cells)")
        axes[i, 1].axis("off")

        # Prediction
        pred_overlay = plot_masks_on_image(img, pred_mask)
        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title(f"Prediction ({np.max(pred_mask)} cells)")
        axes[i, 2].axis("off")

        # Flows
        if flows is not None:
            flow_img = flow_to_rgb(flow)
            axes[i, 3].imshow(flow_img)
            axes[i, 3].set_title("Flow Field")
            axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()


def create_animation(images, interval=200, repeat=True):
    """
    Create animation from image sequence

    Args:
        images (list): List of images
        interval (int): Delay between frames in ms
        repeat (bool): Whether to repeat animation

    Returns:
        matplotlib.animation.ArtistAnimation: Animation object
    """
    from matplotlib.animation import ArtistAnimation

    fig, ax = plt.subplots()

    artists = []
    for img in images:
        artist = ax.imshow(img, animated=True)
        artists.append([artist])

    ax.axis("off")

    anim = ArtistAnimation(fig, artists, interval=interval, repeat=repeat, blit=True)

    return anim


def save_overlay_image(image, masks, save_path, alpha=0.7):
    """
    Save overlay image to file

    Args:
        image (np.ndarray): Background image
        masks (np.ndarray): Segmentation masks
        save_path (str): Output path
        alpha (float): Overlay transparency
    """
    overlay = plot_masks_on_image(image, masks, alpha=alpha)

    # Convert to uint8
    if overlay.max() <= 1:
        overlay = (overlay * 255).astype(np.uint8)

    # Save using opencv (handles RGB->BGR conversion)
    if overlay.shape[-1] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, overlay)


def plot_cell_statistics(masks, image=None):
    """
    Plot statistics about detected cells

    Args:
        masks (np.ndarray): Segmentation masks
        image (np.ndarray): Original image for intensity analysis
    """
    from scipy import ndimage

    # Get cell properties
    unique_labels = np.unique(masks)
    cell_labels = unique_labels[unique_labels > 0]

    if len(cell_labels) == 0:
        print("No cells detected")
        return

    # Calculate areas
    areas = []
    intensities = []

    for label in cell_labels:
        cell_mask = masks == label
        area = np.sum(cell_mask)
        areas.append(area)

        if image is not None:
            intensity = np.mean(image[cell_mask])
            intensities.append(intensity)

    # Plot statistics
    n_plots = 2 if image is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(12, 4))
    if n_plots == 1:
        axes = [axes]

    # Area distribution
    axes[0].hist(areas, bins=30, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Cell Area (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Cell Size Distribution (n={len(areas)})")
    axes[0].grid(True, alpha=0.3)

    # Add statistics text
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    axes[0].axvline(
        mean_area,
        color="red",
        linestyle="--",
        label=f"Mean: {mean_area:.1f}±{std_area:.1f}",
    )
    axes[0].legend()

    # Intensity distribution
    if image is not None:
        axes[1].hist(intensities, bins=30, alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Mean Cell Intensity")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Cell Intensity Distribution")
        axes[1].grid(True, alpha=0.3)

        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        axes[1].axvline(
            mean_intensity,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_intensity:.1f}±{std_intensity:.1f}",
        )
        axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"Cell Detection Summary:")
    print(f"  Total cells: {len(cell_labels)}")
    print(f"  Mean area: {np.mean(areas):.1f} ± {np.std(areas):.1f} pixels")
    print(f"  Area range: {np.min(areas)} - {np.max(areas)} pixels")
    if image is not None:
        print(
            f"  Mean intensity: {np.mean(intensities):.1f} ± {np.std(intensities):.1f}"
        )
