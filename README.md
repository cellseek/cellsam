# CellSAM - Standalone Cell Segmentation with Segment Anything Model

A standalone implementation of CellSAM (Cell Segmentation with Segment Anything Model) for automatic cell segmentation in microscopy images. This implementation is independent and does not require the original cellpose package.

## Features

- **Cell Segmentation**: Automated cell detection and segmentation in 2D and 3D images
- **SAM Integration**: Uses modified Segment Anything Model architecture for robust segmentation
- **Flow Fields**: Predicts flow fields for accurate instance segmentation
- **3D Support**: Full 3D segmentation capabilities
- **Training Pipeline**: Complete training framework for custom datasets
- **Visualization**: Comprehensive plotting and analysis tools
- **Multiple Formats**: Supports various image formats (TIFF, PNG, JPEG, NPY)

## Installation

### Prerequisites

Python 3.8+ with the following packages:

```bash
pip install torch torchvision
pip install segment-anything
pip install numpy scipy opencv-python
pip install matplotlib scikit-image pillow
pip install tqdm
```

### Install from source

1. Clone or download this repository
2. Install the package:

```bash
cd cellsam
pip install -e .
```

Or install requirements directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from cellsam import CellSAM

# Initialize model
model = CellSAM()

# Load your image
import numpy as np
from cellsam.utils import load_image

image = load_image('path/to/your/image.tif')

# Run segmentation
masks, flows, styles = model.eval(image, diameter=30, channels=[0, 0])

print(f"Found {np.max(masks)} cells")
```

### Visualization

```python
from cellsam.visualization import plot_segmentation, plot_cell_statistics

# Plot results
plot_segmentation(image, masks, flows)

# Analyze cell properties
plot_cell_statistics(masks, image)
```

### 3D Segmentation

```python
# For 3D images
masks_3d, flows_3d, styles = model.eval(
    image_3d,
    diameter=30,
    channels=[0, 0],
    do_3D=True
)
```

### Save Results

```python
from cellsam.utils import save_masks

# Save masks and flows
save_masks(masks, 'results.npy', save_flows=True, flows=flows)
```

## API Reference

### CellSAM Class

Main interface for cell segmentation:

```python
CellSAM(device='cuda', model_dir=None)
```

**Parameters:**

- `device`: Computing device ('cuda', 'cpu', or 'mps')
- `model_dir`: Directory for model weights

**Methods:**

#### eval()

```python
masks, flows, styles = model.eval(
    x,                    # Input image(s)
    batch_size=8,         # Batch size for processing
    channels=[0, 0],      # Channel configuration [cytoplasm, nucleus]
    channel_axis=None,    # Channel axis (auto-detected)
    z_axis=None,          # Z-axis for 3D images
    normalize=True,       # Normalize image intensities
    diameter=30,          # Expected cell diameter
    do_3D=False,          # Enable 3D segmentation
    anisotropy=None,      # Z-axis anisotropy for 3D
    net_avg=False,        # Average multiple networks
    augment=False,        # Test-time augmentation
    tile=True,            # Tile large images
    tile_overlap=0.1,     # Overlap between tiles
    resample=True,        # Resample for optimal diameter
    interp=True,          # Interpolate between models
    flow_threshold=0.4,   # Flow error threshold
    cellprob_threshold=0.0,  # Cell probability threshold
    compute_masks=True    # Generate final masks
)
```

### Utility Functions

#### segment_image()

Convenient function for single image segmentation:

```python
from cellsam import segment_image

masks = segment_image(image, diameter=30)
```

#### segment_3d()

For 3D image segmentation:

```python
from cellsam import segment_3d

masks_3d = segment_3d(image_3d, diameter=30)
```

## Training Custom Models

### Prepare Training Data

```python
# Your training images and ground truth masks
train_images = [...]  # List of numpy arrays
train_masks = [...]   # List of ground truth masks

# Optional: validation data
val_images = [...]
val_masks = [...]
```

### Train Model

```python
from cellsam.train import train_model

model, history = train_model(
    train_images, train_masks,
    val_images, val_masks,
    epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    save_dir='./trained_models'
)
```

### Advanced Training

```python
from cellsam.train import CellSAMTrainer, create_data_loaders
from cellsam.model import CellSAMNet

# Create model
model = CellSAMNet()

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_images, train_masks, val_images, val_masks,
    batch_size=4, image_size=512
)

# Create trainer
trainer = CellSAMTrainer(model, device='cuda', learning_rate=1e-4)

# Train
trainer.train(train_loader, val_loader, epochs=100)
```

## Model Architecture

CellSAM uses a modified Segment Anything Model (SAM) architecture:

- **Backbone**: Vision Transformer (ViT) with custom patch size
- **Decoder**: Modified to output flow fields and cell probabilities
- **Flow Fields**: 2D vector fields pointing toward cell centers
- **Instance Segmentation**: Uses flow-based dynamics to separate touching cells

### Model Types

- **cyto**: General cytoplasm model
- **cyto2**: Improved cytoplasm model
- **cyto3**: Latest cytoplasm model (recommended)
- **nuclei**: Nuclei-specific model

## File Formats

### Supported Input Formats

- TIFF/TIF (including multi-page and multi-channel)
- PNG, JPEG, BMP
- NumPy arrays (.npy, .npz)

### Output Formats

- NumPy arrays (.npy) - recommended for masks
- TIFF (.tif) - for visualization
- PNG/JPEG - for 2D visualizations

## Performance Tips

1. **GPU Usage**: Use CUDA for faster processing
2. **Image Size**: Optimal size is 512x512 pixels
3. **Tiling**: Enable tiling for large images (>1024x1024)
4. **Batch Processing**: Process multiple images together
5. **3D Segmentation**: Use `do_3D=True` for volumetric data

## Examples

See `example.py` for complete usage examples including:

- Basic 2D segmentation
- 3D segmentation
- Training interface
- Visualization

Run the example:

```bash
python example.py
```

## Contributing

This is a standalone implementation designed for educational and research purposes. For the official CellPose implementation, please visit the original repository.

## License

This implementation follows the same principles as the original CellPose project. Please check licensing requirements for the Segment Anything Model components.

## Citation

If you use this implementation, please cite the original CellPose papers:

```
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).
Cellpose: a generalist algorithm for cellular segmentation.
Nature methods, 18(1), 100-106.

Pachitariu, M. & Stringer, C. (2022).
Cellpose 2.0: how to train your own model.
Nature methods, 19(12), 1634-1641.
```

And the SAM paper:

```
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023).
Segment anything.
arXiv preprint arXiv:2304.02643.
```
