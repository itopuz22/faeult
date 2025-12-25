#!/usr/bin/env python3
"""
PatchCore Model Training Script for Windows PC

This script trains the PatchCore anomaly detection model using images of
normal (good) parts. The trained model can then be deployed to the
Raspberry Pi for production use.

Usage:
    python training.py --data ./good_images --output model.pkl
    python training.py --data ./good_images --output model.pkl --visualize
    python training.py --data ./good_images --validate ./test_images

Requirements:
    pip install numpy Pillow torch torchvision tqdm matplotlib
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not installed. Progress bars will be simplified.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Visualization disabled.")


# Import the PatchCore model from dariusz.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dariusz import PatchCoreModel, DefectType


class TrainingConfig:
    """Configuration for model training"""

    def __init__(
        self,
        data_dir: str,
        output_path: str = 'patchcore_model.pkl',
        f_coreset: float = 0.1,
        input_size: int = 224,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
        device: str = 'auto',
        validation_dir: Optional[str] = None,
        validation_split: float = 0.1,
        visualize: bool = False,
        report_path: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.output_path = output_path
        self.f_coreset = f_coreset
        self.input_size = input_size
        self.image_extensions = image_extensions
        self.validation_dir = validation_dir
        self.validation_split = validation_split
        self.visualize = visualize
        self.report_path = report_path

        # Auto-detect device
        if device == 'auto':
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if self.device == 'cuda':
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"Using GPU: {gpu_name}")
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"Training configuration:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Input size: {self.input_size}x{self.input_size}")
        logger.info(f"  Coreset ratio: {self.f_coreset}")


def collect_images(directory: str, extensions: Tuple[str, ...]) -> List[str]:
    """Collect all image files from a directory"""
    images = []
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    for ext in extensions:
        images.extend([str(p) for p in directory.glob(f'*{ext}')])
        images.extend([str(p) for p in directory.glob(f'*{ext.upper()}')])

    # Also search subdirectories
    for ext in extensions:
        images.extend([str(p) for p in directory.rglob(f'*{ext}')])
        images.extend([str(p) for p in directory.rglob(f'*{ext.upper()}')])

    # Remove duplicates while preserving order
    seen = set()
    unique_images = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)

    return sorted(unique_images)


def validate_images(image_paths: List[str]) -> Tuple[List[str], List[str]]:
    """Validate that images can be opened and are valid"""
    valid = []
    invalid = []

    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append(path)
        except Exception as e:
            logger.warning(f"Invalid image {path}: {e}")
            invalid.append(path)

    return valid, invalid


def create_progress_bar(iterable, total, desc):
    """Create a progress bar (uses tqdm if available)"""
    if HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc)
    else:
        # Simple fallback progress
        def progress_wrapper(items):
            for i, item in enumerate(items):
                if (i + 1) % max(1, total // 10) == 0 or i == total - 1:
                    print(f"\r{desc}: {i + 1}/{total} ({(i + 1) * 100 // total}%)", end='')
                yield item
            print()
        return progress_wrapper(iterable)


class ModelTrainer:
    """Handles PatchCore model training with progress tracking and validation"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = PatchCoreModel(
            device=config.device,
            f_coreset=config.f_coreset,
            input_size=config.input_size
        )
        self.training_stats: Dict[str, Any] = {}

    def train(self) -> bool:
        """Execute the full training pipeline"""
        logger.info("=" * 60)
        logger.info("PatchCore Model Training")
        logger.info("=" * 60)

        start_time = time.time()

        # Step 1: Collect images
        logger.info(f"\nStep 1: Collecting images from {self.config.data_dir}")
        image_paths = collect_images(
            self.config.data_dir,
            self.config.image_extensions
        )
        logger.info(f"Found {len(image_paths)} images")

        if len(image_paths) == 0:
            logger.error("No images found in the specified directory!")
            return False

        if len(image_paths) < 10:
            logger.warning("Very few training images. Consider adding more for better results.")

        # Step 2: Validate images
        logger.info("\nStep 2: Validating images...")
        valid_paths, invalid_paths = validate_images(image_paths)
        logger.info(f"Valid: {len(valid_paths)}, Invalid: {len(invalid_paths)}")

        if len(valid_paths) == 0:
            logger.error("No valid images found!")
            return False

        # Step 3: Split for validation if needed
        train_paths = valid_paths
        val_paths = []

        if self.config.validation_dir:
            logger.info(f"\nUsing separate validation directory: {self.config.validation_dir}")
            val_paths = collect_images(
                self.config.validation_dir,
                self.config.image_extensions
            )
            val_paths, _ = validate_images(val_paths)
        elif self.config.validation_split > 0:
            split_idx = int(len(valid_paths) * (1 - self.config.validation_split))
            np.random.shuffle(valid_paths)
            train_paths = valid_paths[:split_idx]
            val_paths = valid_paths[split_idx:]
            logger.info(f"\nSplit: {len(train_paths)} training, {len(val_paths)} validation")

        # Step 4: Initialize model
        logger.info("\nStep 3: Initializing feature extractor...")
        if not self.model.initialize():
            logger.warning("Running in simulation mode (PyTorch not available)")

        # Step 5: Train model
        logger.info(f"\nStep 4: Training on {len(train_paths)} images...")
        logger.info(f"Coreset sampling ratio: {self.config.f_coreset}")
        logger.info(f"Device: {self.config.device}")

        def progress_callback(current, total):
            if HAS_TQDM:
                pass  # tqdm handles this
            elif current % max(1, total // 10) == 0 or current == total:
                print(f"\rExtracting features: {current}/{total}", end='')

        try:
            success = self.model.train(train_paths, progress_callback)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

        if not success:
            logger.error("Training failed!")
            return False

        training_time = time.time() - start_time

        # Step 6: Validate if validation set available
        validation_results = None
        if val_paths:
            logger.info(f"\nStep 5: Validating on {len(val_paths)} images...")
            validation_results = self._validate(val_paths)

        # Step 7: Save model
        logger.info(f"\nStep 6: Saving model to {self.config.output_path}")
        self.model.save_model(self.config.output_path)

        # Compile statistics
        self.training_stats = {
            'training_time_seconds': training_time,
            'training_images': len(train_paths),
            'validation_images': len(val_paths),
            'invalid_images': len(invalid_paths),
            'memory_bank_size': self.model.memory_bank.shape[0] if self.model.memory_bank is not None else 0,
            'feature_dim': self.model.memory_bank.shape[1] if self.model.memory_bank is not None else 0,
            'spatial_size': self.model.spatial_size,
            'input_size': self.config.input_size,
            'threshold': self.model.threshold,
            'score_mean': self.model.score_mean,
            'score_std': self.model.score_std,
            'f_coreset': self.config.f_coreset,
            'device': self.config.device,
            'model_path': self.config.output_path,
            'timestamp': datetime.now().isoformat()
        }

        if validation_results:
            self.training_stats['validation'] = validation_results

        # Print summary
        self._print_summary()

        # Generate report
        if self.config.report_path:
            self._save_report()

        # Visualize if requested
        if self.config.visualize and HAS_MATPLOTLIB:
            self._visualize_training()

        return True

    def _validate(self, val_paths: List[str]) -> Dict[str, Any]:
        """Validate model on a set of images"""
        scores = []
        results = []

        for path in create_progress_bar(val_paths, len(val_paths), "Validating"):
            try:
                result = self.model.infer(path, generate_heatmap=False)
                scores.append(result.anomaly_score)
                results.append({
                    'path': path,
                    'score': result.anomaly_score,
                    'is_anomaly': result.is_anomaly,
                    'defect_type': result.defect_type.value
                })
            except Exception as e:
                logger.warning(f"Validation failed for {path}: {e}")

        if not scores:
            return {}

        scores = np.array(scores)

        # For normal images, we expect low false positive rate
        false_positives = sum(1 for r in results if r['is_anomaly'])
        fp_rate = false_positives / len(results) * 100

        return {
            'count': len(scores),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'false_positive_rate': fp_rate,
            'false_positives': false_positives,
            'detailed_results': results
        }

    def _print_summary(self):
        """Print training summary"""
        stats = self.training_stats

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nModel Architecture:")
        print(f"  - Backbone:           WideResNet50-2")
        print(f"  - Feature layers:     layer2 (512ch) + layer3 (1024ch)")
        print(f"  - Feature dimension:  {stats.get('feature_dim', 1536)}")
        print(f"  - Input size:         {stats.get('input_size', 224)}x{stats.get('input_size', 224)}")
        print(f"  - Spatial size:       {stats.get('spatial_size', 28)}x{stats.get('spatial_size', 28)} ({stats.get('spatial_size', 28)**2} patches/image)")

        print(f"\nTraining Statistics:")
        print(f"  - Training images:    {stats['training_images']}")
        print(f"  - Training time:      {stats['training_time_seconds']:.1f} seconds")
        print(f"  - Memory bank size:   {stats['memory_bank_size']} patches")
        print(f"  - Coreset ratio:      {stats['f_coreset']}")
        print(f"  - Device used:        {stats['device']}")

        print(f"\nThreshold Analysis:")
        print(f"  - Threshold:          {stats['threshold']:.4f}")
        print(f"  - Score mean:         {stats['score_mean']:.4f}")
        print(f"  - Score std:          {stats['score_std']:.4f}")

        if 'validation' in stats:
            val = stats['validation']
            print(f"\nValidation Results:")
            print(f"  - Images tested:      {val['count']}")
            print(f"  - Mean score:         {val['mean_score']:.4f}")
            print(f"  - Score range:        [{val['min_score']:.4f}, {val['max_score']:.4f}]")
            print(f"  - False positive rate: {val['false_positive_rate']:.1f}%")

        print(f"\nModel saved to: {stats['model_path']}")
        print("=" * 60)

    def _save_report(self):
        """Save training report to JSON file"""
        report_path = self.config.report_path
        if not report_path.endswith('.json'):
            report_path += '.json'

        # Remove detailed results for cleaner report (can be large)
        stats_copy = self.training_stats.copy()
        if 'validation' in stats_copy and 'detailed_results' in stats_copy['validation']:
            stats_copy['validation'] = {
                k: v for k, v in stats_copy['validation'].items()
                if k != 'detailed_results'
            }

        with open(report_path, 'w') as f:
            json.dump(stats_copy, f, indent=2)

        logger.info(f"Training report saved to: {report_path}")

    def _visualize_training(self):
        """Visualize training results"""
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Training score distribution
        if self.model.train_scores:
            ax1 = axes[0]
            ax1.hist(self.model.train_scores, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(self.model.threshold, color='r', linestyle='--',
                       label=f'Threshold: {self.model.threshold:.4f}')
            ax1.set_xlabel('Anomaly Score')
            ax1.set_ylabel('Count')
            ax1.set_title('Training Score Distribution')
            ax1.legend()

        # Plot 2: Validation scores if available
        if 'validation' in self.training_stats and 'detailed_results' in self.training_stats['validation']:
            ax2 = axes[1]
            val_scores = [r['score'] for r in self.training_stats['validation']['detailed_results']]
            colors = ['red' if r['is_anomaly'] else 'green'
                     for r in self.training_stats['validation']['detailed_results']]

            ax2.scatter(range(len(val_scores)), val_scores, c=colors, alpha=0.6)
            ax2.axhline(self.model.threshold, color='r', linestyle='--',
                       label=f'Threshold: {self.model.threshold:.4f}')
            ax2.set_xlabel('Image Index')
            ax2.set_ylabel('Anomaly Score')
            ax2.set_title('Validation Scores (green=OK, red=flagged)')
            ax2.legend()

        plt.tight_layout()

        # Save figure
        fig_path = self.config.output_path.replace('.pkl', '_training_viz.png')
        plt.savefig(fig_path, dpi=150)
        logger.info(f"Visualization saved to: {fig_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Train PatchCore anomaly detection model (WideResNet50-2 backbone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python training.py --data ./good_images --output model.pkl

  # Training with validation
  python training.py --data ./good_images --validate ./test_good_images

  # Training with visualization
  python training.py --data ./good_images --visualize --report training_report

  # Adjust coreset ratio for memory/accuracy tradeoff
  python training.py --data ./good_images --f-coreset 0.05  # Less memory, faster
  python training.py --data ./good_images --f-coreset 0.2   # More accurate

  # Use GPU for faster training
  python training.py --data ./good_images --device cuda

Model Architecture:
  - Backbone: WideResNet50-2 pretrained on ImageNet
  - Feature layers: layer2 (512ch) + layer3 (1024ch) = 1536 dimensions
  - Patches per image: 784 (28x28 grid for 224x224 input)
        """
    )

    parser.add_argument('--data', '-d', required=True,
                       help='Directory containing training images (good/normal parts)')
    parser.add_argument('--output', '-o', default='patchcore_model.pkl',
                       help='Output path for trained model (default: patchcore_model.pkl)')
    parser.add_argument('--validate', '-v', type=str,
                       help='Directory with validation images (optional)')
    parser.add_argument('--validation-split', type=float, default=0.0,
                       help='Fraction of training data for validation (0-1)')
    parser.add_argument('--f-coreset', type=float, default=0.1,
                       help='Coreset sampling ratio (default: 0.1)')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use for training (default: auto)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization after training')
    parser.add_argument('--report', type=str,
                       help='Save training report to JSON file')

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        data_dir=args.data,
        output_path=args.output,
        f_coreset=args.f_coreset,
        input_size=args.input_size,
        device=args.device,
        validation_dir=args.validate,
        validation_split=args.validation_split,
        visualize=args.visualize,
        report_path=args.report
    )

    # Run training
    trainer = ModelTrainer(config)
    success = trainer.train()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
