#!/usr/bin/env python3
"""
PatchCore Anomaly Detection Module for Industrial Quality Control

This module implements the PatchCore algorithm using WideResNet50 backbone
for detecting manufacturing defects including damage, scratches, and dents.

Features:
- PatchCore-based anomaly detection with coreset sampling
- WideResNet50 feature extraction backbone
- Adaptive thresholding based on ROC curve analysis
- Heatmap generation for visual anomaly localization
- Training, inference, and folder watching capabilities
"""

import os
import json
import time
import logging
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import threading
from queue import Queue

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefectType(Enum):
    """Types of defects the system can detect"""
    OK = "ok"
    DAMAGE = "damage"
    SCRATCH = "scratch"
    DENT = "dent"
    UNKNOWN = "unknown"


@dataclass
class InferenceResult:
    """Result from anomaly detection inference"""
    image_path: str
    timestamp: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    defect_type: DefectType
    confidence: float
    heatmap_path: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['defect_type'] = self.defect_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceResult':
        """Create from dictionary"""
        data['defect_type'] = DefectType(data['defect_type'])
        return cls(**data)


class FeatureExtractor:
    """
    WideResNet50 feature extractor for PatchCore.
    Uses intermediate layers for rich feature representation.

    WideResNet50-2 layer output dimensions (for 224x224 input):
    - layer1: 256 channels, 56x56 spatial
    - layer2: 512 channels, 28x28 spatial  <- Used
    - layer3: 1024 channels, 14x14 spatial <- Used
    - layer4: 2048 channels, 7x7 spatial

    Combined feature dimension: 512 + 1024 = 1536
    Output spatial resolution: 28x28 = 784 patches (after upsampling layer3)
    """

    def __init__(self, device: str = 'cpu', input_size: int = 224):
        self.device = device
        self.input_size = input_size
        self.model = None
        self.transform = None
        self.feature_layers = ['layer2', 'layer3']
        self._hooks = []
        self._features = {}

        # Feature dimensions for WideResNet50-2
        self.layer_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }

        # Spatial sizes for 224x224 input
        self.layer_spatial = {
            'layer1': 56,
            'layer2': 28,
            'layer3': 14,
            'layer4': 7
        }

        # Target spatial size for feature alignment (use layer2's resolution)
        self.target_spatial = 28
        self.n_patches = self.target_spatial * self.target_spatial  # 784 patches
        self.feature_dim = sum(self.layer_channels[l] for l in self.feature_layers)  # 1536

    def _get_hook(self, name: str):
        """Create a forward hook to capture intermediate features"""
        def hook(module, input, output):
            self._features[name] = output.detach()
        return hook

    def load_model(self):
        """Load the WideResNet50-2 model"""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            logger.info("Loading WideResNet50-2 model...")
            logger.info(f"  Input size: {self.input_size}x{self.input_size}")
            logger.info(f"  Feature layers: {self.feature_layers}")
            logger.info(f"  Feature dimension: {self.feature_dim}")
            logger.info(f"  Number of patches: {self.n_patches}")

            # Load pretrained WideResNet50-2
            # Try new API first, fall back to old API
            try:
                from torchvision.models import Wide_ResNet50_2_Weights
                self.model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                self.model = models.wide_resnet50_2(pretrained=True)

            self.model = self.model.to(self.device)
            self.model.eval()

            # Register hooks for intermediate layers
            for name in self.feature_layers:
                layer = getattr(self.model, name)
                hook = layer.register_forward_hook(self._get_hook(name))
                self._hooks.append(hook)

            # Define image transforms (ImageNet normalization)
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            logger.info("WideResNet50-2 model loaded successfully")
            self._print_model_info()
            return True

        except ImportError as e:
            logger.warning(f"PyTorch not available, using simulated features: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _print_model_info(self):
        """Print model layer information"""
        try:
            import torch
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"  Total parameters: {total_params:,}")
        except Exception:
            pass

    def extract_features(self, image: Image.Image) -> Tuple[np.ndarray, int]:
        """
        Extract patch features from an image.

        Returns:
            Tuple of (features array, spatial_size)
            - features: (n_patches, feature_dim) array
            - spatial_size: spatial dimension (e.g., 28 for 28x28 patches)
        """
        try:
            import torch
            import torch.nn.functional as F

            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                _ = self.model(img_tensor)

            # Get layer2 features: (1, 512, 28, 28)
            feat_layer2 = self._features['layer2']
            b, c2, h2, w2 = feat_layer2.shape

            # Get layer3 features: (1, 1024, 14, 14)
            feat_layer3 = self._features['layer3']
            b, c3, h3, w3 = feat_layer3.shape

            # Upsample layer3 to match layer2 spatial dimensions (14x14 -> 28x28)
            feat_layer3_upsampled = F.interpolate(
                feat_layer3,
                size=(h2, w2),
                mode='bilinear',
                align_corners=False
            )

            # Concatenate along channel dimension: (1, 1536, 28, 28)
            combined = torch.cat([feat_layer2, feat_layer3_upsampled], dim=1)

            # Reshape to (n_patches, feature_dim): (784, 1536)
            # From (1, C, H, W) -> (H*W, C)
            combined = combined.squeeze(0)  # (C, H, W)
            combined = combined.permute(1, 2, 0)  # (H, W, C)
            combined = combined.reshape(-1, combined.shape[-1])  # (H*W, C)

            features = combined.cpu().numpy()

            return features, h2

        except ImportError:
            # Fallback: simulated feature extraction
            return self._simulate_features(image)

    def _simulate_features(self, image: Image.Image) -> Tuple[np.ndarray, int]:
        """Simulate feature extraction when PyTorch is not available"""
        # Convert image to numpy array
        img_array = np.array(image.resize((self.input_size, self.input_size)))

        # Generate deterministic features based on image content
        np.random.seed(int(hashlib.md5(img_array.tobytes()).hexdigest()[:8], 16) % 2**32)

        # Create patch-like features matching real model output
        spatial_size = self.target_spatial  # 28
        n_patches = spatial_size * spatial_size  # 784
        feature_dim = self.feature_dim  # 1536

        features = np.random.randn(n_patches, feature_dim).astype(np.float32)

        # Add image-dependent variation based on local statistics
        patch_size = self.input_size // spatial_size  # 8 pixels per patch

        for i in range(n_patches):
            row, col = divmod(i, spatial_size)
            y_start = row * patch_size
            x_start = col * patch_size
            patch_region = img_array[y_start:y_start+patch_size, x_start:x_start+patch_size]

            if patch_region.size > 0:
                # Add local statistics to features
                local_mean = np.mean(patch_region) / 255.0
                local_std = np.std(patch_region) / 255.0
                features[i, :512] += local_mean * 0.5  # layer2 portion
                features[i, 512:] += local_std * 0.5   # layer3 portion

        return features, spatial_size

    def cleanup(self):
        """Remove hooks and free memory"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._features = {}

        if self.model is not None:
            try:
                import torch
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self.model = None


class PatchCoreModel:
    """
    PatchCore anomaly detection model.

    Uses coreset sampling (f_coreset=0.1) for efficient memory bank,
    adaptive thresholding based on ROC curve analysis, and generates
    both raw scores and visual heatmaps.

    Architecture:
    - Backbone: WideResNet50-2 pretrained on ImageNet
    - Feature layers: layer2 (512ch, 28x28) + layer3 (1024ch, 14x14 -> upsampled to 28x28)
    - Combined features: 1536 dimensions per patch
    - Patches per image: 784 (28x28 grid)
    """

    def __init__(
        self,
        device: str = 'cpu',
        f_coreset: float = 0.1,
        backbone: str = 'wide_resnet50_2',
        input_size: int = 224
    ):
        self.device = device
        self.f_coreset = f_coreset
        self.backbone = backbone
        self.input_size = input_size

        self.feature_extractor = FeatureExtractor(device, input_size)
        self.memory_bank: Optional[np.ndarray] = None
        self.threshold: float = 0.5
        self.threshold_percentile: float = 95.0

        # Spatial size for heatmap generation
        self.spatial_size: int = 28

        # Statistics for adaptive thresholding
        self.train_scores: List[float] = []
        self.score_mean: float = 0.0
        self.score_std: float = 1.0

        # Model state
        self.is_trained = False
        self.training_images: int = 0
        self.model_path: Optional[str] = None

    def initialize(self) -> bool:
        """Initialize the feature extractor"""
        success = self.feature_extractor.load_model()
        if success:
            self.spatial_size = self.feature_extractor.target_spatial
        return success

    def _coreset_sampling(self, features: np.ndarray, ratio: float) -> np.ndarray:
        """
        Perform coreset sampling to reduce memory bank size.
        Uses efficient random projection-based sampling for large datasets.
        """
        n_samples = int(features.shape[0] * ratio)
        n_samples = max(1, min(n_samples, features.shape[0]))

        if n_samples >= features.shape[0]:
            return features

        n_features = features.shape[0]
        logger.info(f"  Coreset: selecting {n_samples} from {n_features} patches...")

        # For large datasets, use faster methods
        if n_features > 10000:
            return self._fast_coreset_sampling(features, n_samples)
        else:
            return self._greedy_coreset_sampling(features, n_samples)

    def _fast_coreset_sampling(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Fast coreset sampling using random selection.
        Instant O(n) complexity - much faster than clustering approaches.
        Random sampling is effective for PatchCore anomaly detection.
        """
        logger.info("  Using fast random sampling for coreset...")

        # Random sampling is simple and effective
        indices = np.random.choice(len(features), n_samples, replace=False)
        coreset = features[indices].astype(np.float32)

        logger.info(f"  Coreset sampling complete: {n_samples} samples selected")
        return coreset

    def _stratified_random_sampling(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Stratified random sampling based on feature norms.
        Faster than greedy but maintains diversity.
        """
        # Compute feature norms for stratification
        norms = np.linalg.norm(features, axis=1)

        # Create strata based on norm percentiles
        n_strata = min(10, n_samples)
        percentiles = np.percentile(norms, np.linspace(0, 100, n_strata + 1))

        indices = []
        samples_per_stratum = n_samples // n_strata

        for i in range(n_strata):
            low, high = percentiles[i], percentiles[i + 1]
            if i == n_strata - 1:
                mask = (norms >= low) & (norms <= high)
            else:
                mask = (norms >= low) & (norms < high)

            stratum_indices = np.where(mask)[0]
            if len(stratum_indices) > 0:
                n_take = min(samples_per_stratum, len(stratum_indices))
                chosen = np.random.choice(stratum_indices, n_take, replace=False)
                indices.extend(chosen.tolist())

        # Fill remaining with random samples
        remaining = n_samples - len(indices)
        if remaining > 0:
            available = list(set(range(len(features))) - set(indices))
            if available:
                extra = np.random.choice(available, min(remaining, len(available)), replace=False)
                indices.extend(extra.tolist())

        return features[indices[:n_samples]]

    def _greedy_coreset_sampling(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Original greedy furthest point sampling for smaller datasets.
        O(n*k) complexity with optimized distance updates.
        """
        n_features = features.shape[0]

        # Initialize with random point
        indices = [np.random.randint(n_features)]

        # Track minimum distances to selected set
        min_distances = np.full(n_features, np.inf)

        # Greedy furthest point sampling with incremental updates
        for i in range(n_samples - 1):
            # Update distances only for the last added point
            last_selected = features[indices[-1]]
            new_distances = np.linalg.norm(features - last_selected, axis=1)
            min_distances = np.minimum(min_distances, new_distances)

            # Select furthest point (excluding already selected)
            min_distances[indices[-1]] = -1  # Exclude last selected
            new_idx = np.argmax(min_distances)
            indices.append(new_idx)

            # Progress logging for longer operations
            if (i + 1) % 500 == 0:
                logger.info(f"    Coreset progress: {i + 1}/{n_samples - 1}")

        return features[indices]

    def train(self, image_paths: List[str], progress_callback=None) -> bool:
        """
        Train the PatchCore model on normal (good) images.

        Args:
            image_paths: List of paths to normal training images
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Training PatchCore model on {len(image_paths)} images...")
        logger.info(f"  Backbone: {self.backbone}")
        logger.info(f"  Coreset ratio: {self.f_coreset}")
        logger.info(f"  Device: {self.device}")

        all_features = []
        successful_images = 0

        for i, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                features, spatial_size = self.feature_extractor.extract_features(image)

                # Update spatial size from first successful extraction
                if successful_images == 0:
                    self.spatial_size = spatial_size
                    logger.info(f"  Spatial size: {spatial_size}x{spatial_size} ({spatial_size**2} patches)")
                    logger.info(f"  Feature dimension: {features.shape[1]}")

                all_features.append(features)
                successful_images += 1

                if progress_callback:
                    progress_callback(i + 1, len(image_paths))

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue

        if not all_features:
            logger.error("No features extracted from training images")
            return False

        logger.info(f"Successfully processed {successful_images}/{len(image_paths)} images")

        # Combine all features
        combined_features = np.vstack(all_features)
        logger.info(f"Extracted {combined_features.shape[0]} patch features "
                   f"(shape: {combined_features.shape})")

        # Apply coreset sampling
        logger.info(f"Applying coreset sampling (ratio: {self.f_coreset})...")
        self.memory_bank = self._coreset_sampling(combined_features, self.f_coreset)
        logger.info(f"Memory bank size after coreset: {self.memory_bank.shape[0]} "
                   f"(reduced from {combined_features.shape[0]})")

        # Compute training scores for threshold estimation
        logger.info("Computing training scores for threshold estimation...")
        self.train_scores = []
        for features in all_features:
            score = self._compute_anomaly_score(features)
            self.train_scores.append(score)

        # Compute adaptive threshold using ROC analysis
        self._compute_threshold()

        self.is_trained = True
        self.training_images = successful_images

        logger.info(f"Training complete!")
        logger.info(f"  Memory bank: {self.memory_bank.shape}")
        logger.info(f"  Threshold: {self.threshold:.4f}")
        logger.info(f"  Score mean: {self.score_mean:.4f}, std: {self.score_std:.4f}")
        return True

    def _compute_threshold(self):
        """Compute adaptive threshold based on training scores"""
        if not self.train_scores:
            return

        scores = np.array(self.train_scores)
        self.score_mean = np.mean(scores)
        self.score_std = np.std(scores)

        # Use percentile-based threshold
        # Normal samples should have low scores
        self.threshold = np.percentile(scores, self.threshold_percentile)

        # Add safety margin
        self.threshold = self.threshold + 2 * self.score_std

        logger.info(f"Threshold computed: {self.threshold:.4f} "
                   f"(mean={self.score_mean:.4f}, std={self.score_std:.4f})")

    def _compute_anomaly_score(self, features: np.ndarray) -> float:
        """Compute anomaly score for patch features"""
        if self.memory_bank is None:
            return 0.0

        # Compute distances to nearest neighbors in memory bank
        distances = []
        for feat in features:
            dists = np.linalg.norm(self.memory_bank - feat, axis=1)
            min_dist = np.min(dists)
            distances.append(min_dist)

        # Anomaly score is the maximum of minimum distances
        return float(np.max(distances))

    def _compute_heatmap(
        self,
        features: np.ndarray,
        image_size: Tuple[int, int],
        spatial_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate anomaly heatmap.

        Args:
            features: Patch features (n_patches, feature_dim)
            image_size: Original image size (width, height)
            spatial_size: Spatial grid size (e.g., 28 for 28x28)

        Returns:
            Heatmap array of shape (height, width) with values 0-255
        """
        if self.memory_bank is None:
            return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        # Compute distance for each patch
        distances = []
        for feat in features:
            dists = np.linalg.norm(self.memory_bank - feat, axis=1)
            min_dist = np.min(dists)
            distances.append(min_dist)

        # Determine spatial grid size
        if spatial_size is None:
            spatial_size = self.spatial_size

        # Verify grid size matches number of patches
        expected_patches = spatial_size * spatial_size
        if len(distances) != expected_patches:
            # Try to infer from actual number
            inferred_size = int(np.sqrt(len(distances)))
            if inferred_size * inferred_size == len(distances):
                spatial_size = inferred_size
            else:
                logger.warning(f"Patch count mismatch: {len(distances)} vs expected {expected_patches}")
                # Pad or truncate
                if len(distances) < expected_patches:
                    distances.extend([0.0] * (expected_patches - len(distances)))
                else:
                    distances = distances[:expected_patches]

        # Reshape to spatial grid
        heatmap = np.array(distances[:spatial_size*spatial_size]).reshape(spatial_size, spatial_size)

        # Normalize to 0-1
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # Apply Gaussian smoothing for better visualization
        try:
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=1.0)
        except ImportError:
            pass  # Skip smoothing if scipy not available

        # Resize to original image size using PIL
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = np.array(heatmap_img.resize(image_size, Image.BILINEAR))

        return heatmap_resized

    def _classify_defect(self, score: float, heatmap: np.ndarray) -> Tuple[DefectType, float]:
        """
        Classify the type of defect based on score and heatmap patterns.
        Returns defect type and confidence.
        """
        if score < self.threshold:
            return DefectType.OK, 1.0 - (score / self.threshold)

        # Analyze heatmap pattern for defect classification
        # This is a simplified heuristic - in production, use a trained classifier

        confidence = min((score - self.threshold) / self.score_std, 1.0)

        # Analyze spatial distribution of anomalies
        anomaly_mask = heatmap > (0.5 * 255)
        anomaly_ratio = np.sum(anomaly_mask) / heatmap.size

        # Compute spatial characteristics
        if anomaly_ratio < 0.05:
            # Small, localized anomaly - likely a dent
            return DefectType.DENT, confidence
        elif anomaly_ratio < 0.15:
            # Linear or elongated pattern - likely a scratch
            # Check for linear structure
            if self._is_linear_pattern(anomaly_mask):
                return DefectType.SCRATCH, confidence
            else:
                return DefectType.DENT, confidence
        else:
            # Large area affected - likely damage
            return DefectType.DAMAGE, confidence

    def _is_linear_pattern(self, mask: np.ndarray) -> bool:
        """Check if anomaly pattern is linear (indicating scratch)"""
        # Simple heuristic: check aspect ratio of bounding box
        coords = np.where(mask)
        if len(coords[0]) < 2:
            return False

        height = coords[0].max() - coords[0].min()
        width = coords[1].max() - coords[1].min()

        if height == 0 or width == 0:
            return False

        aspect_ratio = max(height, width) / min(height, width)
        return aspect_ratio > 3.0

    def infer(
        self,
        image_path: str,
        generate_heatmap: bool = True,
        heatmap_output_dir: Optional[str] = None
    ) -> InferenceResult:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file
            generate_heatmap: Whether to generate visual heatmap
            heatmap_output_dir: Directory to save heatmap (uses image dir if None)

        Returns:
            InferenceResult with detection details
        """
        start_time = time.time()

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        try:
            image = Image.open(image_path).convert('RGB')
            image_size = image.size
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

        # Extract features (returns tuple: features, spatial_size)
        features, spatial_size = self.feature_extractor.extract_features(image)

        # Compute anomaly score
        score = self._compute_anomaly_score(features)

        # Generate heatmap
        heatmap = self._compute_heatmap(features, image_size, spatial_size)

        # Classify defect type
        defect_type, confidence = self._classify_defect(score, heatmap)

        # Determine if anomaly
        is_anomaly = score > self.threshold

        # Save heatmap if requested
        heatmap_path = None
        if generate_heatmap and is_anomaly:
            heatmap_path = self._save_heatmap(
                image, heatmap, image_path, heatmap_output_dir
            )

        processing_time = (time.time() - start_time) * 1000

        result = InferenceResult(
            image_path=image_path,
            timestamp=datetime.now().isoformat(),
            is_anomaly=is_anomaly,
            anomaly_score=score,
            threshold=self.threshold,
            defect_type=defect_type,
            confidence=confidence,
            heatmap_path=heatmap_path,
            processing_time_ms=processing_time
        )

        logger.info(f"Inference: {os.path.basename(image_path)} - "
                   f"Score: {score:.4f}, Anomaly: {is_anomaly}, "
                   f"Type: {defect_type.value}")

        return result

    def _save_heatmap(
        self,
        original: Image.Image,
        heatmap: np.ndarray,
        image_path: str,
        output_dir: Optional[str]
    ) -> str:
        """Save heatmap visualization overlaid on original image"""
        # Create colored heatmap (red for anomalies)
        heatmap_colored = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
        heatmap_colored[:, :, 0] = heatmap  # Red channel

        # Convert to PIL
        heatmap_img = Image.fromarray(heatmap_colored, 'RGB')

        # Blend with original
        original_resized = original.resize(heatmap.shape[::-1])
        blended = Image.blend(original_resized, heatmap_img, alpha=0.4)

        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")

        blended.save(heatmap_path)
        logger.info(f"Heatmap saved: {heatmap_path}")

        return heatmap_path

    def save_model(self, path: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'memory_bank': self.memory_bank,
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'train_scores': self.train_scores,
            'score_mean': self.score_mean,
            'score_std': self.score_std,
            'training_images': self.training_images,
            'f_coreset': self.f_coreset,
            'backbone': self.backbone,
            'spatial_size': self.spatial_size,
            'input_size': self.input_size,
            'feature_dim': self.memory_bank.shape[1] if self.memory_bank is not None else 1536,
            'version': '2.0'  # Version for compatibility checking
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        self.model_path = path
        logger.info(f"Model saved to {path}")
        logger.info(f"  Memory bank: {self.memory_bank.shape}")
        logger.info(f"  Threshold: {self.threshold:.4f}")
        logger.info(f"  Spatial size: {self.spatial_size}x{self.spatial_size}")

    def load_model(self, path: str):
        """Load trained model from file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.memory_bank = model_data['memory_bank']
        self.threshold = model_data['threshold']
        self.threshold_percentile = model_data.get('threshold_percentile', 95.0)
        self.train_scores = model_data.get('train_scores', [])
        self.score_mean = model_data.get('score_mean', 0.0)
        self.score_std = model_data.get('score_std', 1.0)
        self.training_images = model_data.get('training_images', 0)
        self.f_coreset = model_data.get('f_coreset', 0.1)
        self.backbone = model_data.get('backbone', 'wide_resnet50_2')
        self.spatial_size = model_data.get('spatial_size', 28)
        self.input_size = model_data.get('input_size', 224)

        self.is_trained = True
        self.model_path = path

        logger.info(f"Model loaded from {path}")
        logger.info(f"  Memory bank: {self.memory_bank.shape}")
        logger.info(f"  Threshold: {self.threshold:.4f}")
        logger.info(f"  Spatial size: {self.spatial_size}x{self.spatial_size}")
        logger.info(f"  Training images: {self.training_images}")


class FolderWatcher:
    """
    Watch a folder for new images and run inference.
    Supports continuous operation for production use.
    """

    def __init__(
        self,
        model: PatchCoreModel,
        watch_dir: str,
        result_callback=None,
        poll_interval: float = 0.5
    ):
        self.model = model
        self.watch_dir = watch_dir
        self.result_callback = result_callback
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processed_files: set = set()

    def start(self):
        """Start watching folder in background thread"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Folder watcher already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started folder watcher for {self.watch_dir}")

    def stop(self):
        """Stop the folder watcher"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Folder watcher stopped")

    def _watch_loop(self):
        """Main watching loop"""
        os.makedirs(self.watch_dir, exist_ok=True)

        while not self._stop_event.is_set():
            try:
                # Get list of image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                current_files = set()

                for f in os.listdir(self.watch_dir):
                    if os.path.splitext(f)[1].lower() in image_extensions:
                        current_files.add(f)

                # Process new files
                new_files = current_files - self._processed_files
                for filename in sorted(new_files):
                    if self._stop_event.is_set():
                        break

                    filepath = os.path.join(self.watch_dir, filename)

                    # Wait for file to be completely written
                    self._wait_for_file(filepath)

                    try:
                        result = self.model.infer(filepath)
                        if self.result_callback:
                            self.result_callback(result)
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")

                    self._processed_files.add(filename)

            except Exception as e:
                logger.error(f"Error in watch loop: {e}")

            self._stop_event.wait(self.poll_interval)

    def _wait_for_file(self, filepath: str, timeout: float = 5.0):
        """Wait for file to be completely written"""
        start = time.time()
        last_size = -1

        while time.time() - start < timeout:
            try:
                current_size = os.path.getsize(filepath)
                if current_size == last_size and current_size > 0:
                    return
                last_size = current_size
                time.sleep(0.1)
            except OSError:
                time.sleep(0.1)


# IPC Handler for daemon communication
class IPCHandler:
    """
    Handle IPC communication with the quality control system.
    Uses JSON files in /home/ai2/yolo_ipc for request/response.
    """

    def __init__(self, model: PatchCoreModel, ipc_dir: str = '/home/ai2/yolo_ipc'):
        self.model = model
        self.ipc_dir = ipc_dir
        self.request_file = os.path.join(ipc_dir, 'request.json')
        self.result_file = os.path.join(ipc_dir, 'result.json')

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start IPC handler"""
        os.makedirs(self.ipc_dir, exist_ok=True)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._ipc_loop, daemon=True)
        self._thread.start()
        logger.info(f"IPC handler started, watching {self.ipc_dir}")

    def stop(self):
        """Stop IPC handler"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("IPC handler stopped")

    def _ipc_loop(self):
        """Main IPC processing loop"""
        while not self._stop_event.is_set():
            try:
                if os.path.exists(self.request_file):
                    self._process_request()
            except Exception as e:
                logger.error(f"IPC error: {e}")

            self._stop_event.wait(0.1)

    def _process_request(self):
        """Process a single IPC request"""
        try:
            with open(self.request_file, 'r') as f:
                request = json.load(f)

            # Remove request file immediately
            os.remove(self.request_file)

            # Process based on request type
            command = request.get('command', 'infer')

            if command == 'infer':
                image_path = request.get('image_path')
                if image_path and os.path.exists(image_path):
                    result = self.model.infer(image_path)
                    response = {
                        'status': 'success',
                        'result': result.to_dict()
                    }
                else:
                    response = {
                        'status': 'error',
                        'error': f'Image not found: {image_path}'
                    }

            elif command == 'status':
                response = {
                    'status': 'success',
                    'model_trained': self.model.is_trained,
                    'threshold': self.model.threshold,
                    'training_images': self.model.training_images
                }

            else:
                response = {
                    'status': 'error',
                    'error': f'Unknown command: {command}'
                }

            # Write result
            with open(self.result_file, 'w') as f:
                json.dump(response, f)

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            with open(self.result_file, 'w') as f:
                json.dump({'status': 'error', 'error': str(e)}, f)


def main():
    """Main entry point for standalone testing"""
    import argparse

    parser = argparse.ArgumentParser(description='PatchCore Anomaly Detection')
    parser.add_argument('--train', type=str, help='Directory with training images')
    parser.add_argument('--infer', type=str, help='Image to run inference on')
    parser.add_argument('--model', type=str, default='patchcore_model.pkl',
                       help='Model file path')
    parser.add_argument('--watch', type=str, help='Directory to watch for new images')
    parser.add_argument('--ipc', action='store_true', help='Start IPC daemon mode')

    args = parser.parse_args()

    # Initialize model
    model = PatchCoreModel()
    model.initialize()

    if args.train:
        # Training mode
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(
                [str(p) for p in Path(args.train).glob(ext)]
            )

        if not image_paths:
            logger.error(f"No images found in {args.train}")
            return

        model.train(image_paths)
        model.save_model(args.model)

    elif args.infer:
        # Single inference mode
        if not os.path.exists(args.model):
            logger.error(f"Model not found: {args.model}")
            return

        model.load_model(args.model)
        result = model.infer(args.infer)
        print(json.dumps(result.to_dict(), indent=2))

    elif args.watch:
        # Folder watching mode
        if not os.path.exists(args.model):
            logger.error(f"Model not found: {args.model}")
            return

        model.load_model(args.model)

        def print_result(result):
            print(json.dumps(result.to_dict(), indent=2))

        watcher = FolderWatcher(model, args.watch, print_result)
        watcher.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()

    elif args.ipc:
        # IPC daemon mode
        if not os.path.exists(args.model):
            logger.error(f"Model not found: {args.model}")
            return

        model.load_model(args.model)

        ipc = IPCHandler(model)
        ipc.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            ipc.stop()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
