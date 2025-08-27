# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unified Anomalib Inference Script.

This script provides a unified interface to perform inference with different anomaly detection models
using their trained checkpoints. It can process single images or entire folders.
"""

import argparse
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2 as cv
import numpy as np
import yaml
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine

# Fix for PyTorch 2.6+ weights_only default change
# Monkey patch torch.load to always use weights_only=False
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False to avoid pickle security issues with OmegaConf
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def instantiate_class_from_config(class_config: dict) -> Any:
    """Instantiate a class from its configuration.

    Args:
        class_config: Configuration dictionary with 'class_path' and 'init_args' keys.

    Returns:
        Instantiated class object.
    """
    class_path = class_config['class_path']
    init_args = class_config.get('init_args', {})

    # Split module and class name
    module_path, class_name = class_path.rsplit('.', 1)

    # Import the module and get the class
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Instantiate the class with init_args
    return cls(**init_args)


def extract_model_name_from_path(class_path: str) -> str:
    """Extract model name from class path.

    Args:
        class_path: Class path like 'anomalib.models.Cfa'

    Returns:
        Model name like 'Cfa'
    """
    return class_path.split('.')[-1]


def load_and_preprocess_image(image_path: Union[str, Path]) -> Image.Image:
    """Load and preprocess a single image.

    Args:
        image_path: Path to the image file.

    Returns:
        Preprocessed PIL Image.
    """
    return Image.open(image_path).convert("RGB")


def collect_image_paths(input_path: Union[str, Path]) -> List[Path]:
    """Collect image paths from input (single file or directory).

    Args:
        input_path: Path to image file or directory containing images.

    Returns:
        List of image file paths.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        # Single image file
        if input_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            return [input_path]
        else:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")

    elif input_path.is_dir():
        # Directory of images
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))

        if not image_paths:
            raise ValueError(f"No image files found in directory: {input_path}")

        return sorted(image_paths)

    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def resize_mask(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize anomaly mask to target size.

    Args:
        mask: Input mask array.
        target_size: Target size (width, height).

    Returns:
        Resized mask.
    """
    return cv.resize(mask, target_size, interpolation=cv.INTER_NEAREST)


def create_overlay_visualization(original_image: Image.Image,
                                anomaly_mask: np.ndarray,
                                output_path: Union[str, Path]) -> None:
    """Create and save overlay visualization of anomaly detection results.

    Args:
        original_image: Original PIL Image.
        anomaly_mask: Anomaly mask array (should be in 0-255 range).
        output_path: Path to save the overlay image.
    """
    # Convert PIL image to numpy array
    img_array = np.array(original_image)

    # Resize mask to match original image size
    resized_mask = resize_mask(anomaly_mask, original_image.size)

    # Ensure mask is in the correct range (0-255)
    if resized_mask.max() <= 1.0:
        resized_mask = (resized_mask * 255).astype(np.uint8)
    else:
        resized_mask = resized_mask.astype(np.uint8)

    # Apply colormap to the mask
    anomaly_map_colored = cv.applyColorMap(resized_mask, cv.COLORMAP_JET)

    # Convert BGR to RGB (OpenCV uses BGR by default)
    anomaly_map_colored = cv.cvtColor(anomaly_map_colored, cv.COLOR_BGR2RGB)

    # Create overlay
    overlay = cv.addWeighted(img_array, 0.6, anomaly_map_colored, 0.4, 0)

    # Save the result
    cv.imwrite(str(output_path), cv.cvtColor(overlay, cv.COLOR_RGB2BGR))


def find_checkpoint_path(model_name: str, category: str, version: int = None) -> Path:
    """Find the checkpoint path for a given model and category.

    Args:
        model_name: Name of the model (e.g., 'Cfa', 'Patchcore').
        category: Dataset category (e.g., 'leather', 'oranges').
        version: Specific version number. If None, finds the latest version.

    Returns:
        Path to the checkpoint file.
    """
    base_path = Path(f"results/{model_name}/MVTecAD/{category}")

    if not base_path.exists():
        raise FileNotFoundError(f"No results found for {model_name}/{category}")

    # Find version directories
    version_dirs = [d for d in base_path.iterdir()
                   if d.is_dir() and d.name.startswith('v') and d.name[1:].isdigit()]

    if not version_dirs:
        raise FileNotFoundError(f"No version directories found in {base_path}")

    if version is not None:
        # Use specific version
        version_path = base_path / f"v{version}"
        if not version_path.exists():
            raise FileNotFoundError(f"Version v{version} not found for {model_name}/{category}")
    else:
        # Find the latest non-empty version directory
        version_dirs.sort(key=lambda d: int(d.name[1:]), reverse=True)
        version_path = None

        for dir_path in version_dirs:
            # Check if directory has any content (files or subdirectories)
            if any(dir_path.iterdir()):
                version_path = dir_path
                break

        if version_path is None:
            raise FileNotFoundError(f"No non-empty version directories found in {base_path}")

    # Look for checkpoint file
    weights_path = version_path / "weights"

    # Check for different possible checkpoint locations
    possible_paths = [
        weights_path / "lightning" / "model.ckpt",
        weights_path / "model.ckpt",
        weights_path / "last.ckpt",
    ]

    for ckpt_path in possible_paths:
        if ckpt_path.exists():
            return ckpt_path

    raise FileNotFoundError(f"No checkpoint file found in {weights_path}")


def determine_output_directory(args, model_name: str, ckpt_path: Path) -> Path:
    """Determine the output directory for inference results.

    Args:
        args: Parsed command line arguments.
        model_name: Name of the model.
        ckpt_path: Path to the checkpoint file.

    Returns:
        Path to the output directory.
    """
    if args.output_dir:
        return Path(args.output_dir)

    # Auto-generate output directory based on checkpoint path
    if not args.checkpoint_path:
        # We used auto-detection, so we know the structure
        version_match = None
        for part in ckpt_path.parts:
            if part.startswith('v') and part[1:].isdigit():
                version_match = part
                break

        if version_match and args.category:
            return Path(f"results/{model_name}/MVTecAD/{args.category}/{version_match}/inference_results")

    # Fallback to default directory
    return Path("inference_results")


def main():
    """Main inference function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Perform inference using trained anomaly detection models"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="uflow.yaml",
        help="Path to model YAML configuration file (e.g., 'cfa.yaml' or path to specific file)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/data/projects/anii-anomalias/anomalib_training/datasets/MVTecAD/leather/test/cut",
        help="Path to input image file or directory containing images"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to model checkpoint file. If not provided, will auto-detect from results directory"
    )
    parser.add_argument(
        "--category",
        type=str,
        default='leather',
        help="Dataset category (required if checkpoint-path is not provided)"
    )
    parser.add_argument(
        "--version",
        type=int,
        help="Model version number (default: latest)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save inference results (default: auto-generated based on model/category/version)"
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save overlay visualizations of anomaly detection results"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save anomaly masks as separate files"
    )

    args = parser.parse_args()

    # Set working directory to script location
    os.chdir(os.path.dirname(__file__))

    # Resolve model config path
    model_config_path = Path(args.model_config)
    if not model_config_path.exists():
        # Try in the configs directory
        configs_path = Path("configs") / args.model_config
        if configs_path.exists():
            model_config_path = configs_path
        else:
            raise FileNotFoundError(f"Model configuration file not found: {args.model_config}")

    # Load model configuration
    model_config = load_yaml_config(model_config_path)
    print(f"Loaded model config from: {model_config_path}")

    # Extract model information
    model_class_path = model_config['model']['class_path']
    model_name = extract_model_name_from_path(model_class_path)

    # Determine checkpoint path
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
    else:
        if not args.category:
            raise ValueError("Either --checkpoint-path or --category must be provided")
        ckpt_path = find_checkpoint_path(model_name, args.category, args.version)

    print(f"Using checkpoint: {ckpt_path}")

    # Determine output directory
    output_dir = determine_output_directory(args, model_name, ckpt_path)
    print(f"Output directory: {output_dir}")

    # Collect input images
    print(f"Collecting images from: {args.input_path}")
    image_paths = collect_image_paths(args.input_path)
    print(f"Found {len(image_paths)} image(s)")

    # Load and preprocess images
    print("Loading and preprocessing images...")
    images = [load_and_preprocess_image(path) for path in image_paths]

    # Store original image sizes for later use
    original_sizes = [img.size for img in images]

    # Create predict dataset
    dataset = PredictDataset(images=images)

    # Instantiate model
    print(f"Instantiating model: {model_name}")
    model = instantiate_class_from_config(model_config['model'])

    # Create engine - set default_root_dir to our output directory
    # This ensures all saves (both manual and automatic) go to the same location
    engine = Engine(default_root_dir=output_dir)
    print(f"All outputs (manual and automatic) will be saved to: {output_dir}")

    # Print inference configuration
    print("=" * 60)
    print("Inference Configuration:")
    print(f"Model: {model_name} ({model_class_path})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input images: {len(image_paths)}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Perform inference
    print("Starting inference...")
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=ckpt_path,
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process results
    if predictions is not None:
        print("Processing results...")

        for i, prediction in enumerate(predictions):
            image_path = image_paths[i]
            original_image = images[i]
            original_size = original_sizes[i]

            # Extract prediction results
            anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
            pred_label = prediction.pred_label    # Image-level label (0: normal, 1: anomalous)
            pred_score = prediction.pred_score    # Image-level anomaly score
            pred_mask = prediction.pred_mask      # Image-level anomaly mask

            # Convert mask to numpy array and normalize to 0-255 range
            pred_mask_numpy = pred_mask.to_dense().numpy().squeeze()

            # Handle different mask types (boolean, float, etc.)
            if pred_mask_numpy.dtype == bool:
                # Boolean mask: convert True to 255, False to 0
                pred_mask_img = (pred_mask_numpy * 255).astype(np.uint8)
            elif pred_mask_numpy.max() > 0:
                # Numeric mask: normalize to 0-1 range first, then scale to 0-255
                pred_mask_numpy = pred_mask_numpy.astype(np.float32)
                pred_mask_normalized = (pred_mask_numpy - pred_mask_numpy.min()) / (pred_mask_numpy.max() - pred_mask_numpy.min())
                pred_mask_img = (pred_mask_normalized * 255).astype(np.uint8)
            else:
                # Handle case where all values are zero
                pred_mask_img = pred_mask_numpy.astype(np.uint8)

            # Create output filename stem
            output_stem = image_path.stem

            # Print results for this image
            print(f"\nImage: {image_path.name}")
            print(f"  Prediction: {'Anomalous' if pred_label else 'Normal'}")
            print(f"  Score: {pred_score.item():.4f}")
            print(f"  Mask range: {pred_mask_numpy.min():.4f} - {pred_mask_numpy.max():.4f}")

            # Save masks if requested
            if args.save_masks:
                # Save original mask
                mask_path = output_dir / f"{output_stem}_mask.png"
                cv.imwrite(str(mask_path), pred_mask_img)

                # Save resized mask to match original image
                resized_mask = resize_mask(pred_mask_img, original_size)
                resized_mask_path = output_dir / f"{output_stem}_mask_resized.png"
                cv.imwrite(str(resized_mask_path), resized_mask)

                print(f"  Saved masks: {mask_path}, {resized_mask_path}")

            # Save overlay visualization if requested
            if args.save_overlay:
                overlay_path = output_dir / f"{output_stem}_overlay.png"
                create_overlay_visualization(original_image, pred_mask_img, overlay_path)
                print(f"  Saved overlay: {overlay_path}")

        print(f"\nInference completed! Results saved to: {output_dir}")

        # Summary statistics
        anomaly_count = sum(1 for pred in predictions if pred.pred_label)
        normal_count = len(predictions) - anomaly_count
        avg_score = sum(pred.pred_score.item() for pred in predictions) / len(predictions)

        print("\nSummary:")
        print(f"  Total images: {len(predictions)}")
        print(f"  Normal: {normal_count}")
        print(f"  Anomalous: {anomaly_count}")
        print(f"  Average score: {avg_score:.4f}")

    else:
        print("No predictions returned!")


if __name__ == "__main__":
    main()
