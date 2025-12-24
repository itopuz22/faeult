#!/usr/bin/env python3
"""
Main Quality Control Application

Integrates the PatchCore anomaly detection (dariusz.py) with hardware
automation (automation.py) for a complete industrial quality control system.

Features:
- Automatic power failure recovery
- State persistence across restarts
- Real-time statistics and monitoring
- Integration with dashboard via events
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, asdict
import atexit

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dariusz import PatchCoreModel, DefectType, InferenceResult, IPCHandler
from automation import (
    QualityControlSystem, GPIOController, CameraController,
    FTPUploader, IPCClient, FTPConfig
)

# Configure logging
LOG_DIR = '/var/log/qc_system'
DATA_DIR = '/var/lib/qc_system'
STATE_FILE = os.path.join(DATA_DIR, 'system_state.json')
RESULTS_FILE = os.path.join(DATA_DIR, 'inspection_results.json')
MODEL_FILE = os.path.join(DATA_DIR, 'patchcore_model.pkl')
IPC_DIR = '/home/ai2/yolo_ipc'

# Ensure directories exist
for dir_path in [LOG_DIR, DATA_DIR, IPC_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Configure file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'qc_system.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Persistent system state"""
    is_running: bool = False
    total_inspections: int = 0
    ok_count: int = 0
    nok_count: int = 0
    error_count: int = 0
    last_inspection_time: Optional[str] = None
    last_anomaly_time: Optional[str] = None
    session_start_time: Optional[str] = None
    model_loaded: bool = False
    model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        return cls(**data)


class InspectionResultStore:
    """Store for inspection results with persistence"""

    def __init__(self, file_path: str, max_results: int = 1000):
        self.file_path = file_path
        self.max_results = max_results
        self.results: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load results from file"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    self.results = json.load(f)
                logger.info(f"Loaded {len(self.results)} previous results")
        except Exception as e:
            logger.warning(f"Could not load results: {e}")
            self.results = []

    def _save(self):
        """Save results to file"""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.results, f)
        except Exception as e:
            logger.error(f"Could not save results: {e}")

    def add(self, result: Dict[str, Any]):
        """Add a new result"""
        with self._lock:
            self.results.append(result)
            # Trim old results
            if len(self.results) > self.max_results:
                self.results = self.results[-self.max_results:]
            self._save()

    def get_recent(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get most recent results"""
        with self._lock:
            return list(reversed(self.results[-count:]))

    def get_anomalies(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent anomaly detections"""
        with self._lock:
            anomalies = [r for r in self.results if r.get('is_anomaly')]
            return list(reversed(anomalies[-count:]))

    def get_by_id(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        """Get result by inspection ID"""
        with self._lock:
            for result in reversed(self.results):
                if result.get('inspection_id') == inspection_id:
                    return result
            return None

    def update_classification(self, inspection_id: str, new_classification: str) -> bool:
        """Update the classification of an inspection (for operator review)"""
        with self._lock:
            for result in self.results:
                if result.get('inspection_id') == inspection_id:
                    result['operator_classification'] = new_classification
                    result['classification_time'] = datetime.now().isoformat()
                    self._save()
                    return True
            return False


class QualityControlDaemon:
    """
    Main quality control daemon with fault tolerance.

    Provides:
    - Automatic recovery after power failures
    - Persistent state management
    - Real-time event notifications
    - IPC-based AI inference daemon
    """

    def __init__(self):
        self.state = SystemState()
        self.result_store = InspectionResultStore(RESULTS_FILE)

        # AI Model
        self.model = PatchCoreModel()
        self.ipc_handler: Optional[IPCHandler] = None

        # Hardware control
        self.hardware = QualityControlSystem(
            capture_dir='/tmp/qc_captures',
            ipc_dir=IPC_DIR
        )

        # Event callbacks for dashboard
        self._event_callbacks: List[Callable] = []

        # Control flags
        self._shutdown_event = threading.Event()
        self._inspection_lock = threading.Lock()

        # Statistics
        self._session_stats = {
            'inspections': 0,
            'ok': 0,
            'nok': 0,
            'errors': 0,
            'avg_processing_time': 0.0
        }

    def _load_state(self):
        """Load persistent state"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.state = SystemState.from_dict(data)
                logger.info("System state restored from previous session")
                logger.info(f"Previous stats - Total: {self.state.total_inspections}, "
                           f"OK: {self.state.ok_count}, NOK: {self.state.nok_count}")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Save persistent state"""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state.to_dict(), f)
        except Exception as e:
            logger.error(f"Could not save state: {e}")

    def add_event_callback(self, callback: Callable):
        """Add callback for system events (used by dashboard)"""
        self._event_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all registered callbacks"""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def initialize(self) -> bool:
        """Initialize the system"""
        logger.info("="*60)
        logger.info("Initializing Quality Control System")
        logger.info("="*60)

        # Load previous state
        self._load_state()

        # Initialize AI model
        logger.info("Initializing PatchCore model...")
        self.model.initialize()

        # Load trained model if available
        if os.path.exists(MODEL_FILE):
            try:
                self.model.load_model(MODEL_FILE)
                self.state.model_loaded = True
                self.state.model_path = MODEL_FILE
                logger.info("Trained model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                self.state.model_loaded = False
        else:
            logger.warning("No trained model found. System will need training.")
            self.state.model_loaded = False

        # Start IPC handler for AI daemon
        self.ipc_handler = IPCHandler(self.model, IPC_DIR)
        self.ipc_handler.start()
        logger.info("IPC handler started")

        # Initialize hardware
        if not self.hardware.initialize():
            logger.error("Hardware initialization failed")
            return False

        # Set up result callback
        self.hardware.set_result_callback(self._on_inspection_result)

        # Update state
        self.state.session_start_time = datetime.now().isoformat()
        self._save_state()

        # Emit initialization event
        self._emit_event('system_initialized', {
            'model_loaded': self.state.model_loaded,
            'total_inspections': self.state.total_inspections
        })

        logger.info("Quality Control System initialized successfully")
        return True

    def _on_inspection_result(self, result: Dict[str, Any]):
        """Handle inspection result from hardware system"""
        with self._inspection_lock:
            # Update statistics
            self.state.total_inspections += 1
            self.state.last_inspection_time = result.get('timestamp')

            self._session_stats['inspections'] += 1

            if result.get('is_anomaly'):
                self.state.nok_count += 1
                self.state.last_anomaly_time = result.get('timestamp')
                self._session_stats['nok'] += 1
            elif result.get('status') == 'completed':
                self.state.ok_count += 1
                self._session_stats['ok'] += 1
            else:
                self.state.error_count += 1
                self._session_stats['errors'] += 1

            # Update average processing time
            pt = result.get('processing_time_ms', 0)
            n = self._session_stats['inspections']
            self._session_stats['avg_processing_time'] = (
                (self._session_stats['avg_processing_time'] * (n - 1) + pt) / n
            )

            # Store result
            self.result_store.add(result)

            # Save state
            self._save_state()

            # Emit events
            self._emit_event('inspection_complete', result)

            if result.get('is_anomaly'):
                self._emit_event('anomaly_detected', {
                    'inspection_id': result.get('inspection_id'),
                    'defect_type': result.get('defect_type'),
                    'confidence': result.get('confidence'),
                    'image_path': result.get('image_path'),
                    'heatmap_path': result.get('heatmap_path')
                })

    def train_model(self, training_dir: str, progress_callback=None) -> bool:
        """Train the PatchCore model on normal images"""
        logger.info(f"Training model from: {training_dir}")

        # Collect image paths
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend([str(p) for p in Path(training_dir).glob(ext)])

        if not image_paths:
            logger.error(f"No training images found in {training_dir}")
            return False

        logger.info(f"Found {len(image_paths)} training images")

        try:
            success = self.model.train(image_paths, progress_callback)
            if success:
                self.model.save_model(MODEL_FILE)
                self.state.model_loaded = True
                self.state.model_path = MODEL_FILE
                self._save_state()
                logger.info("Model training complete and saved")
                self._emit_event('model_trained', {
                    'training_images': len(image_paths),
                    'threshold': self.model.threshold
                })
            return success
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def start(self):
        """Start the quality control system"""
        if not self.state.model_loaded:
            logger.warning("Model not loaded - system running in limited mode")

        logger.info("Starting continuous inspection mode...")
        self.state.is_running = True
        self._save_state()

        self.hardware.start_continuous_mode()

        self._emit_event('system_started', {
            'timestamp': datetime.now().isoformat()
        })

    def stop(self):
        """Stop the quality control system"""
        logger.info("Stopping quality control system...")
        self.state.is_running = False
        self._save_state()

        self.hardware.stop_continuous_mode()

        self._emit_event('system_stopped', {
            'timestamp': datetime.now().isoformat()
        })

    def trigger_manual_inspection(self) -> Dict[str, Any]:
        """Trigger a manual inspection"""
        logger.info("Manual inspection triggered")
        return self.hardware.run_inspection()

    def get_statistics(self) -> Dict[str, Any]:
        """Get current system statistics"""
        hw_stats = self.hardware.get_statistics()
        return {
            'lifetime': {
                'total_inspections': self.state.total_inspections,
                'ok_count': self.state.ok_count,
                'nok_count': self.state.nok_count,
                'error_count': self.state.error_count
            },
            'session': self._session_stats,
            'hardware': hw_stats,
            'system': {
                'is_running': self.state.is_running,
                'model_loaded': self.state.model_loaded,
                'session_start': self.state.session_start_time,
                'last_inspection': self.state.last_inspection_time,
                'last_anomaly': self.state.last_anomaly_time
            }
        }

    def get_recent_results(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent inspection results"""
        return self.result_store.get_recent(count)

    def get_anomalies(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent anomaly detections"""
        return self.result_store.get_anomalies(count)

    def classify_result(self, inspection_id: str, classification: str) -> bool:
        """Operator classification of a result"""
        success = self.result_store.update_classification(inspection_id, classification)
        if success:
            self._emit_event('result_classified', {
                'inspection_id': inspection_id,
                'classification': classification
            })
        return success

    def shutdown(self):
        """Clean shutdown of the system"""
        logger.info("Shutting down Quality Control System...")
        self._shutdown_event.set()

        self.stop()

        if self.ipc_handler:
            self.ipc_handler.stop()

        self.hardware.cleanup()

        self._save_state()

        logger.info("System shutdown complete")


# Global daemon instance
_daemon: Optional[QualityControlDaemon] = None


def get_daemon() -> QualityControlDaemon:
    """Get or create the global daemon instance"""
    global _daemon
    if _daemon is None:
        _daemon = QualityControlDaemon()
    return _daemon


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    if _daemon:
        _daemon.shutdown()
    sys.exit(0)


def main():
    """Main entry point for the QC daemon"""
    import argparse

    parser = argparse.ArgumentParser(description='Quality Control System')
    parser.add_argument('--train', type=str, help='Training directory')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--single', action='store_true', help='Run single inspection')
    parser.add_argument('--status', action='store_true', help='Show system status')

    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Get daemon instance
    daemon = get_daemon()

    # Initialize
    if not daemon.initialize():
        logger.error("Initialization failed")
        sys.exit(1)

    # Register cleanup
    atexit.register(daemon.shutdown)

    if args.train:
        # Training mode
        def progress(current, total):
            print(f"\rTraining: {current}/{total} ({current*100//total}%)", end='')

        success = daemon.train_model(args.train, progress)
        print()  # New line after progress
        sys.exit(0 if success else 1)

    elif args.single:
        # Single inspection
        result = daemon.trigger_manual_inspection()
        print(json.dumps(result, indent=2))
        daemon.shutdown()

    elif args.status:
        # Show status
        stats = daemon.get_statistics()
        print(json.dumps(stats, indent=2))
        daemon.shutdown()

    elif args.daemon:
        # Daemon mode
        logger.info("Starting in daemon mode...")
        daemon.start()

        try:
            while not daemon._shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        daemon.shutdown()

    else:
        parser.print_help()
        daemon.shutdown()


if __name__ == '__main__':
    main()
