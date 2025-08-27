# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unified Anomalib Training Script with YAML Configuration.

This script provides a unified interface to train different anomaly detection models
using YAML configuration files from the anomalib examples directory.
"""

import argparse
import os
import importlib
from pathlib import Path
from typing import Any, Dict

import yaml
from jsonargparse import ArgumentParser, Namespace
from omegaconf import DictConfig, OmegaConf

from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.utils.config import update_config


def get_next_experiment_number(model_name: str, category: str) -> int:
    """Get the next experiment number for the given model and category.
    
    Args:
        model_name: Name of the model (e.g., 'Cfa', 'Uflow', 'Patchcore').
        category: Dataset category (e.g., 'oranges', 'leather').
        
    Returns:
        Next available experiment number.
    """
    base_path = Path(f"results/{model_name}/MVTecAD/{category}")
    
    if not base_path.exists():
        return 0
    
    # Find existing experiment directories (format: v{number})
    existing_dirs = [d for d in base_path.iterdir() 
                    if d.is_dir() and d.name.startswith('v') and d.name[1:].isdigit()]
    
    if not existing_dirs:
        return 0
    
    # Check for empty directories first - reuse the lowest numbered empty one
    existing_dirs.sort(key=lambda d: int(d.name[1:]))  # Sort by experiment number
    for dir_path in existing_dirs:
        # Check if directory is empty (no files or subdirectories)
        if not any(dir_path.iterdir()):
            return int(dir_path.name[1:])
    
    # If no empty directories found, get the highest experiment number and add 1
    max_exp_num = max(int(d.name[1:]) for d in existing_dirs)
    return max_exp_num + 1


def create_experiment_paths(results_path: str, model_name: str, category: str, exp_num: int) -> dict:
    """Create experiment directory structure and return paths.
    
    Args:
        results_path: Base results directory path.
        model_name: Name of the model.
        category: Dataset category.
        exp_num: Experiment number.
        
    Returns:
        Dictionary containing all relevant paths.
    """
    main_path = Path(f"{results_path}/{model_name}/MVTecAD/{category}/v{exp_num}")
    
    paths = {
        'main': main_path,
        'weights': main_path / "weights",
        'logs': main_path / "logs",
        'results': Path(results_path)
    }
    
    return paths


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


def merge_configs(model_config: dict, trainer_config: dict = None, category: str = None) -> dict:
    """Merge model and trainer configurations with fixed MVTecAD data configuration.
    
    Args:
        model_config: Model configuration dictionary.
        trainer_config: Trainer configuration dictionary (optional).
        category: Dataset category to override (optional).
        
    Returns:
        Merged configuration dictionary.
    """
    merged_config = {
        'model': model_config.get('model', model_config),
        'trainer': trainer_config or model_config.get('trainer', {'max_epochs': 30})
    }
    
    # Determine batch size based on model type
    model_class_path = merged_config['model'].get('class_path', '')
    model_name = extract_model_name_from_path(model_class_path)
    
    # EfficientAd requires batch size of 1
    if model_name == 'EfficientAd':
        train_batch_size = 1
        eval_batch_size = 1
    else:
        train_batch_size = 8
        eval_batch_size = 16
    
    # Fixed MVTecAD configuration
    merged_config['data'] = {
        'class_path': 'anomalib.data.MVTecAD',
        'init_args': {
            'root': './datasets/MVTecAD',
            'category': category,
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
        }
    }
    
    return merged_config


def extract_model_name_from_path(class_path: str) -> str:
    """Extract model name from class path.
    
    Args:
        class_path: Class path like 'anomalib.models.Cfa'
        
    Returns:
        Model name like 'Cfa'
    """
    return class_path.split('.')[-1]


def create_engine_with_callbacks(trainer_config: dict, paths: dict, 
                                enable_checkpointing: bool = True,
                                enable_tensorboard: bool = True) -> Engine:
    """Create training engine with callbacks based on configuration.
    
    Args:
        trainer_config: Trainer configuration dictionary.
        paths: Dictionary containing experiment paths.
        enable_checkpointing: Whether to enable model checkpointing.
        enable_tensorboard: Whether to enable TensorBoard logging.
        
    Returns:
        Initialized Engine object.
    """
    callbacks = []
    
    # Add existing callbacks from trainer config
    existing_callbacks = trainer_config.get('callbacks', [])
    for callback_config in existing_callbacks:
        if isinstance(callback_config, dict) and 'class_path' in callback_config:
            callback = instantiate_class_from_config(callback_config)
            callbacks.append(callback)
    
    # Add checkpoint callback if enabled
    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(paths['weights']),
            every_n_epochs=1,  # Save every epoch to capture progress
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        print(f"Checkpoint callback configured: {paths['weights']}")
    
    # Create TensorBoard logger if enabled
    logger = None
    if enable_tensorboard:
        logger = AnomalibTensorBoardLogger(
            save_dir=str(paths['logs']),
        )
        print(f"TensorBoard logger configured: {paths['logs']}")
    
    # Extract trainer parameters (excluding callbacks which we handle separately)
    trainer_params = {k: v for k, v in trainer_config.items() if k != 'callbacks'}
    
    print(f"Engine configuration:")
    print(f"  - enable_checkpointing: {enable_checkpointing}")
    print(f"  - default_root_dir: {paths['main']}")
    print(f"  - callbacks: {len(callbacks)} callback(s)")
    print(f"  - trainer_params: {trainer_params}")
    
    return Engine(
        callbacks=callbacks if callbacks else None,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        default_root_dir=paths['results'],
        **trainer_params
    )


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models using YAML configuration files"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="cfa.yaml",
        help="Path to model YAML configuration file (e.g., 'cfa.yaml' or path to specific file)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="leather",
        help="Dataset category (default: leather)"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results",
        help="Base path for results (default: results)"
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="Disable model checkpointing"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    
    args = parser.parse_args()
    
    # Set working directory to script location
    os.chdir(os.path.dirname(__file__))
    
    # Resolve model config path
    model_config_path = Path(args.model_config)
    if not model_config_path.exists():
        # Try in the anomalib examples configs directory
        examples_path = Path("configs") / args.model_config
        if examples_path.exists():
            model_config_path = examples_path
        else:
            raise FileNotFoundError(f"Model configuration file not found: {args.model_config}")
    
    # Load model configuration
    model_config = load_yaml_config(model_config_path)
    print(f"Loaded model config from: {model_config_path}")
    
    # Merge configurations (data config is fixed to MVTecAD)
    merged_config = merge_configs(model_config, category=args.category)
    
    # Convert to DictConfig for update_config compatibility
    config_dict = OmegaConf.create(merged_config)
    
    # Apply anomalib's config updates
    updated_config = update_config(config_dict)
    
    # Extract model name for path creation
    model_class_path = updated_config.model.class_path
    model_name = extract_model_name_from_path(model_class_path)
    
    # Get next experiment number and create paths
    exp_num = get_next_experiment_number(model_name, args.category)
    paths = create_experiment_paths(args.results_path, model_name, args.category, exp_num)
    
    # Print configuration summary
    print("=" * 60)
    print("Training Configuration:")
    print(f"Model: {model_name} ({model_class_path})")
    print(f"Category: {args.category}")
    max_epochs = getattr(updated_config.trainer, 'max_epochs', 'Not specified')
    print(f"Max Epochs: {max_epochs}")
    print(f"Experiment: {exp_num}")
    print(f"Main Path: {paths['main']}")
    print(f"Weights Path: {paths['weights']}")
    if not args.no_tensorboard:
        print(f"Logs Path: {paths['logs']}")
    print("=" * 60)
    
    # Instantiate components
    print("Instantiating model...")
    model = instantiate_class_from_config(updated_config.model)
    
    print("Instantiating datamodule...")
    datamodule = instantiate_class_from_config(updated_config.data)
    
    print("Creating engine...")
    engine = create_engine_with_callbacks(
        updated_config.trainer, 
        paths,
        enable_checkpointing=not args.no_checkpointing,
        enable_tensorboard=not args.no_tensorboard
    )
    
    # Train the model
    print("Starting training...")
    engine.fit(datamodule=datamodule, model=model)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
