#!/usr/bin/env python3
"""
PatchCore Model Testing Script for Windows PC

This script tests the trained PatchCore model on images to detect
manufacturing defects (damage, scratches, dents).

Usage:
    python test.py --model model.pkl --image sample.jpg
    python test.py --model model.pkl --dir ./test_images --output results/
    python test.py --model model.pkl --dir ./test_images --threshold 1.5

Requirements:
    pip install numpy Pillow torch torchvision tqdm matplotlib
"""

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

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

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import the PatchCore model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dariusz import PatchCoreModel, InferenceResult, DefectType


@dataclass
class TestConfig:
    """Configuration for testing"""
    model_path: str
    image_path: Optional[str] = None
    directory: Optional[str] = None
    output_dir: Optional[str] = None
    threshold: Optional[float] = None
    generate_heatmaps: bool = True
    save_results: bool = True
    show_results: bool = False
    export_csv: bool = False
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')


class TestResult:
    """Container for test results with analysis"""

    def __init__(self):
        self.results: List[InferenceResult] = []
        self.summary: Dict[str, Any] = {}

    def add(self, result: InferenceResult):
        self.results.append(result)

    def analyze(self) -> Dict[str, Any]:
        """Analyze all results and generate summary statistics"""
        if not self.results:
            return {}

        total = len(self.results)
        anomalies = [r for r in self.results if r.is_anomaly]
        ok_count = total - len(anomalies)

        scores = [r.anomaly_score for r in self.results]
        times = [r.processing_time_ms for r in self.results]

        # Count by defect type
        defect_counts = {}
        for result in anomalies:
            dtype = result.defect_type.value
            defect_counts[dtype] = defect_counts.get(dtype, 0) + 1

        self.summary = {
            'total_images': total,
            'ok_count': ok_count,
            'anomaly_count': len(anomalies),
            'ok_rate': ok_count / total * 100 if total > 0 else 0,
            'anomaly_rate': len(anomalies) / total * 100 if total > 0 else 0,
            'defect_breakdown': defect_counts,
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            },
            'timing_statistics': {
                'mean_ms': float(np.mean(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'total_ms': float(np.sum(times))
            },
            'threshold_used': self.results[0].threshold if self.results else 0
        }

        return self.summary

    def get_anomalies(self) -> List[InferenceResult]:
        """Get all anomaly results"""
        return [r for r in self.results if r.is_anomaly]

    def get_ok(self) -> List[InferenceResult]:
        """Get all OK results"""
        return [r for r in self.results if not r.is_anomaly]


class ModelTester:
    """Handles model testing and result visualization"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.model = PatchCoreModel()
        self.test_results = TestResult()

    def load_model(self) -> bool:
        """Load the trained model"""
        if not os.path.exists(self.config.model_path):
            logger.error(f"Model not found: {self.config.model_path}")
            return False

        try:
            logger.info(f"Loading model from {self.config.model_path}")
            self.model.initialize()
            self.model.load_model(self.config.model_path)

            # Override threshold if specified
            if self.config.threshold is not None:
                logger.info(f"Using custom threshold: {self.config.threshold}")
                self.model.threshold = self.config.threshold

            logger.info(f"Model loaded. Threshold: {self.model.threshold:.4f}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def test_single(self, image_path: str) -> Optional[InferenceResult]:
        """Test a single image"""
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None

        try:
            # Determine heatmap output directory
            heatmap_dir = None
            if self.config.generate_heatmaps:
                if self.config.output_dir:
                    heatmap_dir = os.path.join(self.config.output_dir, 'heatmaps')
                else:
                    heatmap_dir = os.path.join(os.path.dirname(image_path), 'heatmaps')
                os.makedirs(heatmap_dir, exist_ok=True)

            result = self.model.infer(
                image_path,
                generate_heatmap=self.config.generate_heatmaps,
                heatmap_output_dir=heatmap_dir
            )

            self.test_results.add(result)
            return result

        except Exception as e:
            logger.error(f"Failed to test {image_path}: {e}")
            return None

    def test_directory(self, directory: str) -> TestResult:
        """Test all images in a directory"""
        image_paths = self._collect_images(directory)

        if not image_paths:
            logger.error(f"No images found in {directory}")
            return self.test_results

        logger.info(f"Testing {len(image_paths)} images...")

        # Create output directory if specified
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)

        # Process images
        iterator = image_paths
        if HAS_TQDM:
            iterator = tqdm(image_paths, desc="Testing")

        for path in iterator:
            result = self.test_single(path)
            if result and not HAS_TQDM:
                status = "ANOMALY" if result.is_anomaly else "OK"
                print(f"  {os.path.basename(path)}: {status} (score: {result.anomaly_score:.4f})")

        return self.test_results

    def _collect_images(self, directory: str) -> List[str]:
        """Collect image files from directory"""
        images = []
        directory = Path(directory)

        for ext in self.config.image_extensions:
            images.extend([str(p) for p in directory.glob(f'*{ext}')])
            images.extend([str(p) for p in directory.glob(f'*{ext.upper()}')])
            images.extend([str(p) for p in directory.rglob(f'*{ext}')])
            images.extend([str(p) for p in directory.rglob(f'*{ext.upper()}')])

        # Remove duplicates
        return sorted(set(images))

    def print_results(self):
        """Print test results summary"""
        summary = self.test_results.analyze()

        if not summary:
            print("No results to display")
            return

        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        print(f"\nOverall Results:")
        print(f"  Total images tested: {summary['total_images']}")
        print(f"  OK:                  {summary['ok_count']} ({summary['ok_rate']:.1f}%)")
        print(f"  Anomalies:           {summary['anomaly_count']} ({summary['anomaly_rate']:.1f}%)")

        if summary['defect_breakdown']:
            print(f"\nDefect Breakdown:")
            for defect_type, count in summary['defect_breakdown'].items():
                print(f"  - {defect_type.upper()}: {count}")

        print(f"\nScore Statistics:")
        stats = summary['score_statistics']
        print(f"  Mean:    {stats['mean']:.4f}")
        print(f"  Std:     {stats['std']:.4f}")
        print(f"  Min:     {stats['min']:.4f}")
        print(f"  Max:     {stats['max']:.4f}")
        print(f"  Median:  {stats['median']:.4f}")

        print(f"\nTiming:")
        timing = summary['timing_statistics']
        print(f"  Avg per image: {timing['mean_ms']:.1f} ms")
        print(f"  Total time:    {timing['total_ms']/1000:.2f} seconds")

        print(f"\nThreshold used: {summary['threshold_used']:.4f}")
        print("=" * 60)

        # List anomalies
        anomalies = self.test_results.get_anomalies()
        if anomalies:
            print(f"\nDetected Anomalies ({len(anomalies)}):")
            for i, result in enumerate(anomalies[:20], 1):  # Show first 20
                print(f"  {i}. {os.path.basename(result.image_path)}")
                print(f"     Type: {result.defect_type.value}, Score: {result.anomaly_score:.4f}, "
                      f"Confidence: {result.confidence:.1%}")

            if len(anomalies) > 20:
                print(f"  ... and {len(anomalies) - 20} more")

    def save_results(self, output_path: Optional[str] = None):
        """Save results to JSON file"""
        if output_path is None:
            if self.config.output_dir:
                output_path = os.path.join(self.config.output_dir, 'test_results.json')
            else:
                output_path = 'test_results.json'

        summary = self.test_results.analyze()

        output_data = {
            'summary': summary,
            'results': [r.to_dict() for r in self.test_results.results],
            'config': {
                'model_path': self.config.model_path,
                'threshold': self.model.threshold,
                'timestamp': datetime.now().isoformat()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def export_csv(self, output_path: Optional[str] = None):
        """Export results to CSV file"""
        if output_path is None:
            if self.config.output_dir:
                output_path = os.path.join(self.config.output_dir, 'test_results.csv')
            else:
                output_path = 'test_results.csv'

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Image', 'Timestamp', 'Is Anomaly', 'Defect Type',
                'Anomaly Score', 'Threshold', 'Confidence', 'Processing Time (ms)',
                'Heatmap Path'
            ])

            for result in self.test_results.results:
                writer.writerow([
                    os.path.basename(result.image_path),
                    result.timestamp,
                    result.is_anomaly,
                    result.defect_type.value,
                    f"{result.anomaly_score:.4f}",
                    f"{result.threshold:.4f}",
                    f"{result.confidence:.4f}",
                    f"{result.processing_time_ms:.1f}",
                    result.heatmap_path or ''
                ])

        logger.info(f"CSV exported to: {output_path}")

    def visualize_results(self):
        """Visualize test results"""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not installed. Cannot visualize.")
            return

        results = self.test_results.results
        if not results:
            return

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig)

        # Plot 1: Score distribution
        ax1 = fig.add_subplot(gs[0, 0])
        scores = [r.anomaly_score for r in results]
        colors = ['red' if r.is_anomaly else 'green' for r in results]

        ax1.hist([r.anomaly_score for r in results if not r.is_anomaly],
                bins=20, alpha=0.7, label='OK', color='green')
        ax1.hist([r.anomaly_score for r in results if r.is_anomaly],
                bins=20, alpha=0.7, label='Anomaly', color='red')
        ax1.axvline(self.model.threshold, color='black', linestyle='--',
                   label=f'Threshold: {self.model.threshold:.3f}')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Score Distribution')
        ax1.legend()

        # Plot 2: OK vs Anomaly pie chart
        ax2 = fig.add_subplot(gs[0, 1])
        ok_count = len([r for r in results if not r.is_anomaly])
        anomaly_count = len(results) - ok_count
        ax2.pie([ok_count, anomaly_count],
               labels=['OK', 'Anomaly'],
               colors=['green', 'red'],
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title('OK vs Anomaly Ratio')

        # Plot 3: Defect type breakdown
        ax3 = fig.add_subplot(gs[0, 2])
        anomalies = [r for r in results if r.is_anomaly]
        if anomalies:
            defect_types = {}
            for r in anomalies:
                dt = r.defect_type.value
                defect_types[dt] = defect_types.get(dt, 0) + 1

            ax3.bar(defect_types.keys(), defect_types.values(),
                   color=['purple', 'orange', 'cyan', 'magenta'][:len(defect_types)])
            ax3.set_ylabel('Count')
            ax3.set_title('Defect Type Breakdown')
        else:
            ax3.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center')
            ax3.set_title('Defect Type Breakdown')

        # Plot 4: Score vs index (timeline)
        ax4 = fig.add_subplot(gs[1, :2])
        indices = range(len(results))
        scores = [r.anomaly_score for r in results]
        colors = ['red' if r.is_anomaly else 'green' for r in results]

        ax4.scatter(indices, scores, c=colors, alpha=0.6, s=30)
        ax4.axhline(self.model.threshold, color='black', linestyle='--',
                   label=f'Threshold: {self.model.threshold:.3f}')
        ax4.set_xlabel('Image Index')
        ax4.set_ylabel('Anomaly Score')
        ax4.set_title('Scores by Image Order')
        ax4.legend()

        # Plot 5: Processing time
        ax5 = fig.add_subplot(gs[1, 2])
        times = [r.processing_time_ms for r in results]
        ax5.hist(times, bins=20, color='blue', alpha=0.7)
        ax5.axvline(np.mean(times), color='red', linestyle='--',
                   label=f'Mean: {np.mean(times):.1f}ms')
        ax5.set_xlabel('Processing Time (ms)')
        ax5.set_ylabel('Count')
        ax5.set_title('Processing Time Distribution')
        ax5.legend()

        plt.tight_layout()

        # Save figure
        if self.config.output_dir:
            fig_path = os.path.join(self.config.output_dir, 'test_visualization.png')
        else:
            fig_path = 'test_visualization.png'

        plt.savefig(fig_path, dpi=150)
        logger.info(f"Visualization saved to: {fig_path}")

        if self.config.show_results:
            plt.show()

    def show_anomaly_gallery(self, max_images: int = 12):
        """Show gallery of detected anomalies with heatmaps"""
        if not HAS_MATPLOTLIB:
            return

        anomalies = self.test_results.get_anomalies()[:max_images]

        if not anomalies:
            logger.info("No anomalies to display")
            return

        n_images = len(anomalies)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, result in enumerate(anomalies):
            row, col = divmod(idx, cols)
            ax = axes[row, col]

            # Load and display image (prefer heatmap if available)
            img_path = result.heatmap_path or result.image_path
            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')

            ax.set_title(f"{result.defect_type.value.upper()}\n"
                        f"Score: {result.anomaly_score:.3f}",
                        fontsize=10)
            ax.axis('off')

        # Hide empty subplots
        for idx in range(n_images, rows * cols):
            row, col = divmod(idx, cols)
            axes[row, col].axis('off')

        plt.suptitle('Detected Anomalies', fontsize=14)
        plt.tight_layout()

        if self.config.output_dir:
            fig_path = os.path.join(self.config.output_dir, 'anomaly_gallery.png')
        else:
            fig_path = 'anomaly_gallery.png'

        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"Anomaly gallery saved to: {fig_path}")

        if self.config.show_results:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Test PatchCore anomaly detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python test.py --model model.pkl --image sample.jpg

  # Test directory of images
  python test.py --model model.pkl --dir ./test_images

  # Test with custom output directory
  python test.py --model model.pkl --dir ./test_images --output ./results

  # Test with custom threshold
  python test.py --model model.pkl --dir ./test_images --threshold 1.5

  # Test with visualization
  python test.py --model model.pkl --dir ./test_images --visualize --show

  # Export results to CSV
  python test.py --model model.pkl --dir ./test_images --csv
        """
    )

    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained model file (.pkl)')
    parser.add_argument('--image', '-i', type=str,
                       help='Single image to test')
    parser.add_argument('--dir', '-d', type=str,
                       help='Directory of images to test')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for results')
    parser.add_argument('--threshold', '-t', type=float,
                       help='Custom detection threshold (overrides model threshold)')
    parser.add_argument('--no-heatmaps', action='store_true',
                       help='Disable heatmap generation')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--show', '-s', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--csv', action='store_true',
                       help='Export results to CSV')
    parser.add_argument('--gallery', action='store_true',
                       help='Generate anomaly gallery image')

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Either --image or --dir must be specified")

    # Create configuration
    config = TestConfig(
        model_path=args.model,
        image_path=args.image,
        directory=args.dir,
        output_dir=args.output,
        threshold=args.threshold,
        generate_heatmaps=not args.no_heatmaps,
        show_results=args.show
    )

    # Create tester
    tester = ModelTester(config)

    # Load model
    if not tester.load_model():
        sys.exit(1)

    # Run tests
    if args.image:
        # Single image test
        result = tester.test_single(args.image)
        if result:
            print("\n" + "=" * 50)
            print(f"Image: {os.path.basename(args.image)}")
            print(f"Result: {'ANOMALY' if result.is_anomaly else 'OK'}")
            print(f"Defect Type: {result.defect_type.value}")
            print(f"Anomaly Score: {result.anomaly_score:.4f}")
            print(f"Threshold: {result.threshold:.4f}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Processing Time: {result.processing_time_ms:.1f} ms")
            if result.heatmap_path:
                print(f"Heatmap: {result.heatmap_path}")
            print("=" * 50)

    elif args.dir:
        # Directory test
        tester.test_directory(args.dir)
        tester.print_results()

        # Save results
        tester.save_results()

        # Export CSV if requested
        if args.csv:
            tester.export_csv()

        # Generate visualization if requested
        if args.visualize:
            tester.visualize_results()

        # Generate gallery if requested
        if args.gallery:
            tester.show_anomaly_gallery()


if __name__ == '__main__':
    main()
