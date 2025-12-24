#!/usr/bin/env python3
"""
Raspberry Pi Hardware Automation Module for Quality Control System

This module provides hardware control functionality including:
- GPIO control using gpiod library for relay board management
- Raspberry HQ camera integration via Picamera2
- FTP upload to production server
- IPC-based communication with AI inference daemon

Hardware Configuration:
- Input trigger: GPIO 17
- Relay outputs: K1 (GPIO 26/OK), K2 (GPIO 20/Light), K3 (GPIO 21/NOK)
- Camera: Raspberry HQ Camera via Picamera2
- FTP Server: 10.5.191.182 (PROD/elopkcats321!)
"""

import os
import sys
import json
import time
import logging
import threading
import ftplib
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# GPIO Pin Configuration
class GPIOPins:
    """GPIO pin assignments for the quality control system"""
    INPUT_TRIGGER = 17  # Input trigger signal

    # Relay outputs
    K1_OK = 26    # Relay K1 - OK signal (good part)
    K2_LIGHT = 20  # Relay K2 - Lighting control
    K3_NOK = 21   # Relay K3 - NOK signal (bad part)


# FTP Configuration
@dataclass
class FTPConfig:
    """FTP server configuration"""
    host: str = '10.5.191.182'
    username: str = 'PROD'
    password: str = 'elopkcats321!'
    base_path: str = '/A10_AI'
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 2.0


class RelayState(Enum):
    """Relay output states"""
    OFF = 0
    ON = 1


class InspectionResult(Enum):
    """Inspection result types"""
    OK = "ok"
    NOK = "nok"
    PENDING = "pending"
    ERROR = "error"


class GPIOController:
    """
    GPIO controller using gpiod library.
    Manages input triggers and relay outputs.
    """

    def __init__(self, chip_path: str = '/dev/gpiochip4'):
        self.chip_path = chip_path
        self.chip = None
        self.input_line = None
        self.output_lines: Dict[str, Any] = {}

        self._trigger_callback: Optional[Callable] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track relay states
        self._relay_states = {
            'K1_OK': RelayState.OFF,
            'K2_LIGHT': RelayState.OFF,
            'K3_NOK': RelayState.OFF
        }

    def initialize(self) -> bool:
        """Initialize GPIO chip and lines"""
        try:
            import gpiod

            logger.info(f"Initializing GPIO controller with chip {self.chip_path}")

            # Open GPIO chip
            self.chip = gpiod.Chip(self.chip_path)

            # Configure input line (trigger)
            self.input_line = self.chip.get_line(GPIOPins.INPUT_TRIGGER)
            self.input_line.request(
                consumer='qc_system',
                type=gpiod.LINE_REQ_DIR_IN,
                flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP
            )

            # Configure output lines (relays)
            output_pins = {
                'K1_OK': GPIOPins.K1_OK,
                'K2_LIGHT': GPIOPins.K2_LIGHT,
                'K3_NOK': GPIOPins.K3_NOK
            }

            for name, pin in output_pins.items():
                line = self.chip.get_line(pin)
                line.request(
                    consumer='qc_system',
                    type=gpiod.LINE_REQ_DIR_OUT,
                    default_vals=[0]
                )
                self.output_lines[name] = line

            logger.info("GPIO controller initialized successfully")
            return True

        except ImportError:
            logger.warning("gpiod library not available, using simulation mode")
            return self._init_simulation()
        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")
            return self._init_simulation()

    def _init_simulation(self) -> bool:
        """Initialize simulation mode when hardware is not available"""
        logger.info("GPIO controller running in simulation mode")
        self._simulation_mode = True
        return True

    def set_relay(self, relay_name: str, state: RelayState):
        """Set relay output state"""
        if hasattr(self, '_simulation_mode') and self._simulation_mode:
            self._relay_states[relay_name] = state
            logger.info(f"[SIM] Relay {relay_name} set to {state.name}")
            return

        if relay_name in self.output_lines:
            self.output_lines[relay_name].set_value(state.value)
            self._relay_states[relay_name] = state
            logger.info(f"Relay {relay_name} set to {state.name}")
        else:
            logger.warning(f"Unknown relay: {relay_name}")

    def get_relay_state(self, relay_name: str) -> RelayState:
        """Get current relay state"""
        return self._relay_states.get(relay_name, RelayState.OFF)

    def signal_ok(self, duration: float = 0.5):
        """Activate OK signal relay"""
        self.set_relay('K1_OK', RelayState.ON)
        time.sleep(duration)
        self.set_relay('K1_OK', RelayState.OFF)

    def signal_nok(self, duration: float = 0.5):
        """Activate NOK signal relay"""
        self.set_relay('K3_NOK', RelayState.ON)
        time.sleep(duration)
        self.set_relay('K3_NOK', RelayState.OFF)

    def set_light(self, state: bool):
        """Control lighting relay"""
        self.set_relay('K2_LIGHT', RelayState.ON if state else RelayState.OFF)

    def read_trigger(self) -> bool:
        """Read trigger input state"""
        if hasattr(self, '_simulation_mode') and self._simulation_mode:
            return False

        if self.input_line:
            return self.input_line.get_value() == 0  # Active low with pull-up
        return False

    def set_trigger_callback(self, callback: Callable):
        """Set callback function for trigger events"""
        self._trigger_callback = callback

    def start_trigger_monitor(self):
        """Start background thread to monitor trigger input"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._trigger_monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Trigger monitor started")

    def stop_trigger_monitor(self):
        """Stop trigger monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Trigger monitor stopped")

    def _trigger_monitor_loop(self):
        """Monitor trigger input for rising/falling edges"""
        last_state = False

        while not self._stop_event.is_set():
            try:
                current_state = self.read_trigger()

                # Detect rising edge (trigger activated)
                if current_state and not last_state:
                    logger.info("Trigger detected!")
                    if self._trigger_callback:
                        try:
                            self._trigger_callback()
                        except Exception as e:
                            logger.error(f"Trigger callback error: {e}")

                last_state = current_state
                time.sleep(0.01)  # 10ms polling interval

            except Exception as e:
                logger.error(f"Trigger monitor error: {e}")
                time.sleep(0.1)

    def cleanup(self):
        """Release GPIO resources"""
        self.stop_trigger_monitor()

        # Turn off all relays
        for name in self.output_lines:
            self.set_relay(name, RelayState.OFF)

        if self.input_line:
            self.input_line.release()

        for line in self.output_lines.values():
            line.release()

        if self.chip:
            self.chip.close()

        logger.info("GPIO resources released")


class CameraController:
    """
    Camera controller for Raspberry HQ Camera.
    Uses Picamera2 library for capture.
    """

    def __init__(self, capture_dir: str = '/tmp/qc_captures'):
        self.capture_dir = capture_dir
        self.camera = None
        self._is_initialized = False
        self._simulation_mode = False

        # Camera settings
        self.resolution = (4056, 3040)  # Full HQ camera resolution
        self.preview_resolution = (640, 480)

    def initialize(self) -> bool:
        """Initialize the camera"""
        try:
            from picamera2 import Picamera2

            logger.info("Initializing Raspberry HQ Camera...")

            self.camera = Picamera2()

            # Configure for still capture
            config = self.camera.create_still_configuration(
                main={"size": self.resolution},
                lores={"size": self.preview_resolution},
                display="lores"
            )
            self.camera.configure(config)
            self.camera.start()

            # Allow camera to warm up
            time.sleep(2)

            os.makedirs(self.capture_dir, exist_ok=True)

            self._is_initialized = True
            logger.info("Camera initialized successfully")
            return True

        except ImportError:
            logger.warning("Picamera2 not available, using simulation mode")
            return self._init_simulation()
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return self._init_simulation()

    def _init_simulation(self) -> bool:
        """Initialize simulation mode"""
        self._simulation_mode = True
        self._is_initialized = True
        os.makedirs(self.capture_dir, exist_ok=True)
        logger.info("Camera running in simulation mode")
        return True

    def capture(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Capture an image.

        Args:
            filename: Optional filename. If not provided, generates timestamp-based name.

        Returns:
            Path to captured image or None on failure.
        """
        if not self._is_initialized:
            logger.error("Camera not initialized")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"capture_{timestamp}.jpg"

        filepath = os.path.join(self.capture_dir, filename)

        try:
            if self._simulation_mode:
                # Create a simulated image
                return self._create_simulation_image(filepath)

            self.camera.capture_file(filepath)
            logger.info(f"Image captured: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None

    def _create_simulation_image(self, filepath: str) -> str:
        """Create a simulated test image"""
        try:
            from PIL import Image
            import numpy as np

            # Create a test pattern image
            width, height = 640, 480
            img_array = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)

            # Add some structure
            img_array[100:380, 100:540] = 180  # Light rectangle

            # Randomly add defects for testing
            if np.random.random() > 0.7:
                # Add simulated defect
                x, y = np.random.randint(150, 400), np.random.randint(150, 300)
                img_array[y:y+30, x:x+30] = 50

            img = Image.fromarray(img_array, 'RGB')
            img.save(filepath)
            logger.info(f"[SIM] Image created: {filepath}")
            return filepath

        except ImportError:
            # Create empty file as fallback
            with open(filepath, 'wb') as f:
                f.write(b'\x00' * 1024)
            return filepath

    def cleanup(self):
        """Release camera resources"""
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                logger.warning(f"Error closing camera: {e}")

        self._is_initialized = False
        logger.info("Camera resources released")


class FTPUploader:
    """
    FTP uploader for quality control images and results.
    Uploads to /A10_AI/{DDMMYY}/ directory structure.
    """

    def __init__(self, config: Optional[FTPConfig] = None):
        self.config = config or FTPConfig()
        self._connection: Optional[ftplib.FTP] = None
        self._lock = threading.Lock()

    def _connect(self) -> bool:
        """Establish FTP connection"""
        try:
            self._connection = ftplib.FTP()
            self._connection.connect(self.config.host, timeout=self.config.timeout)
            self._connection.login(self.config.username, self.config.password)
            logger.info(f"Connected to FTP server {self.config.host}")
            return True

        except Exception as e:
            logger.error(f"FTP connection failed: {e}")
            self._connection = None
            return False

    def _disconnect(self):
        """Close FTP connection"""
        if self._connection:
            try:
                self._connection.quit()
            except Exception:
                pass
            self._connection = None

    def _ensure_directory(self, path: str):
        """Ensure directory exists on FTP server"""
        if not self._connection:
            return

        parts = path.strip('/').split('/')
        current = ''

        for part in parts:
            current = f"{current}/{part}"
            try:
                self._connection.cwd(current)
            except ftplib.error_perm:
                try:
                    self._connection.mkd(current)
                    self._connection.cwd(current)
                except Exception as e:
                    logger.warning(f"Could not create directory {current}: {e}")

        # Return to root
        self._connection.cwd('/')

    def upload(
        self,
        local_path: str,
        remote_filename: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> bool:
        """
        Upload a file to the FTP server.

        Args:
            local_path: Path to local file
            remote_filename: Optional remote filename (uses local name if not provided)
            subfolder: Optional subfolder within date directory

        Returns:
            True if upload successful
        """
        if not os.path.exists(local_path):
            logger.error(f"File not found: {local_path}")
            return False

        if remote_filename is None:
            remote_filename = os.path.basename(local_path)

        # Build remote path: /A10_AI/{DDMMYY}/{subfolder}/
        date_folder = datetime.now().strftime('%d%m%y')
        remote_dir = f"{self.config.base_path}/{date_folder}"
        if subfolder:
            remote_dir = f"{remote_dir}/{subfolder}"

        with self._lock:
            for attempt in range(self.config.retry_count):
                try:
                    if not self._connection:
                        if not self._connect():
                            continue

                    # Ensure directory exists
                    self._ensure_directory(remote_dir)
                    self._connection.cwd(remote_dir)

                    # Upload file
                    with open(local_path, 'rb') as f:
                        self._connection.storbinary(f'STOR {remote_filename}', f)

                    logger.info(f"Uploaded {local_path} to {remote_dir}/{remote_filename}")
                    self._connection.cwd('/')
                    return True

                except (ftplib.error_temp, socket.timeout, OSError) as e:
                    logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                    self._disconnect()
                    time.sleep(self.config.retry_delay)

                except Exception as e:
                    logger.error(f"Upload failed: {e}")
                    self._disconnect()
                    break

        return False

    def upload_with_metadata(
        self,
        image_path: str,
        metadata: Dict[str, Any],
        result_type: str = 'ok'
    ) -> bool:
        """Upload image with associated metadata JSON"""
        # Upload image
        subfolder = 'OK' if result_type.lower() == 'ok' else 'NOK'
        if not self.upload(image_path, subfolder=subfolder):
            return False

        # Create and upload metadata
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        metadata_path = f"/tmp/{base_name}_meta.json"

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        result = self.upload(metadata_path, subfolder=subfolder)

        # Clean up temp file
        try:
            os.remove(metadata_path)
        except Exception:
            pass

        return result

    def cleanup(self):
        """Close FTP connection"""
        self._disconnect()


class IPCClient:
    """
    IPC client for communication with AI inference daemon.
    Uses JSON files in /home/ai2/yolo_ipc directory.
    """

    def __init__(self, ipc_dir: str = '/home/ai2/yolo_ipc'):
        self.ipc_dir = ipc_dir
        self.request_file = os.path.join(ipc_dir, 'request.json')
        self.result_file = os.path.join(ipc_dir, 'result.json')
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize IPC directory"""
        try:
            os.makedirs(self.ipc_dir, exist_ok=True)
            logger.info(f"IPC client initialized at {self.ipc_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize IPC: {e}")
            return False

    def request_inference(
        self,
        image_path: str,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Send inference request to AI daemon and wait for result.

        Args:
            image_path: Path to image file
            timeout: Maximum time to wait for result

        Returns:
            Inference result dictionary or None on failure
        """
        with self._lock:
            # Clear any existing result
            if os.path.exists(self.result_file):
                os.remove(self.result_file)

            # Write request
            request = {
                'command': 'infer',
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }

            try:
                with open(self.request_file, 'w') as f:
                    json.dump(request, f)

                logger.info(f"Inference request sent for {image_path}")

            except Exception as e:
                logger.error(f"Failed to write request: {e}")
                return None

            # Wait for result
            start_time = time.time()
            while time.time() - start_time < timeout:
                if os.path.exists(self.result_file):
                    try:
                        with open(self.result_file, 'r') as f:
                            result = json.load(f)

                        os.remove(self.result_file)
                        logger.info(f"Inference result received")
                        return result

                    except Exception as e:
                        logger.error(f"Failed to read result: {e}")
                        return None

                time.sleep(0.05)

            logger.warning("Inference request timed out")
            return None

    def get_daemon_status(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get AI daemon status"""
        with self._lock:
            if os.path.exists(self.result_file):
                os.remove(self.result_file)

            request = {
                'command': 'status',
                'timestamp': datetime.now().isoformat()
            }

            try:
                with open(self.request_file, 'w') as f:
                    json.dump(request, f)

            except Exception as e:
                logger.error(f"Failed to write status request: {e}")
                return None

            start_time = time.time()
            while time.time() - start_time < timeout:
                if os.path.exists(self.result_file):
                    try:
                        with open(self.result_file, 'r') as f:
                            result = json.load(f)

                        os.remove(self.result_file)
                        return result

                    except Exception as e:
                        logger.error(f"Failed to read status: {e}")
                        return None

                time.sleep(0.1)

            return None


class QualityControlSystem:
    """
    Main quality control system integrating all components.
    Handles the complete inspection workflow.
    """

    def __init__(
        self,
        capture_dir: str = '/tmp/qc_captures',
        ipc_dir: str = '/home/ai2/yolo_ipc',
        ftp_config: Optional[FTPConfig] = None
    ):
        self.gpio = GPIOController()
        self.camera = CameraController(capture_dir)
        self.ftp = FTPUploader(ftp_config)
        self.ipc = IPCClient(ipc_dir)

        self._is_running = False
        self._inspection_count = 0
        self._ok_count = 0
        self._nok_count = 0
        self._error_count = 0

        self._result_callback: Optional[Callable] = None

    def initialize(self) -> bool:
        """Initialize all subsystems"""
        logger.info("Initializing quality control system...")

        gpio_ok = self.gpio.initialize()
        camera_ok = self.camera.initialize()
        ipc_ok = self.ipc.initialize()

        if gpio_ok and camera_ok and ipc_ok:
            logger.info("Quality control system initialized successfully")
            return True
        else:
            logger.warning("Some subsystems failed to initialize")
            return gpio_ok or camera_ok  # Continue with what we have

    def set_result_callback(self, callback: Callable):
        """Set callback for inspection results"""
        self._result_callback = callback

    def run_inspection(self) -> Dict[str, Any]:
        """
        Run a complete inspection cycle.

        Returns:
            Inspection result dictionary
        """
        self._inspection_count += 1
        inspection_id = f"INS_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._inspection_count}"

        logger.info(f"Starting inspection {inspection_id}")

        result = {
            'inspection_id': inspection_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'is_anomaly': False,
            'defect_type': 'unknown',
            'confidence': 0.0,
            'anomaly_score': 0.0,
            'image_path': None,
            'heatmap_path': None,
            'processing_time_ms': 0.0
        }

        try:
            start_time = time.time()

            # Turn on lighting
            self.gpio.set_light(True)
            time.sleep(0.1)  # Allow light to stabilize

            # Capture image
            image_path = self.camera.capture(f"{inspection_id}.jpg")
            if not image_path:
                raise RuntimeError("Failed to capture image")

            result['image_path'] = image_path

            # Turn off lighting
            self.gpio.set_light(False)

            # Request AI inference
            inference_result = self.ipc.request_inference(image_path)

            if inference_result and inference_result.get('status') == 'success':
                ai_result = inference_result.get('result', {})
                result.update({
                    'status': 'completed',
                    'is_anomaly': ai_result.get('is_anomaly', False),
                    'defect_type': ai_result.get('defect_type', 'unknown'),
                    'confidence': ai_result.get('confidence', 0.0),
                    'anomaly_score': ai_result.get('anomaly_score', 0.0),
                    'heatmap_path': ai_result.get('heatmap_path')
                })
            else:
                # AI daemon not available - log error but continue
                logger.warning("AI inference unavailable, marking as error")
                result['status'] = 'error'
                result['error'] = 'AI inference unavailable'

            result['processing_time_ms'] = (time.time() - start_time) * 1000

            # Signal result via relays
            if result['status'] == 'completed':
                if result['is_anomaly']:
                    self.gpio.signal_nok()
                    self._nok_count += 1
                    result['decision'] = 'NOK'
                else:
                    self.gpio.signal_ok()
                    self._ok_count += 1
                    result['decision'] = 'OK'

                # Upload to FTP
                self.ftp.upload_with_metadata(
                    image_path,
                    result,
                    result['decision']
                )
            else:
                self._error_count += 1
                result['decision'] = 'ERROR'

            # Call result callback
            if self._result_callback:
                try:
                    self._result_callback(result)
                except Exception as e:
                    logger.error(f"Result callback error: {e}")

        except Exception as e:
            logger.error(f"Inspection failed: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
            self._error_count += 1

        logger.info(f"Inspection {inspection_id} completed: {result['status']}")
        return result

    def start_continuous_mode(self):
        """Start continuous inspection mode (trigger-based)"""
        if self._is_running:
            logger.warning("System already running")
            return

        self._is_running = True

        def on_trigger():
            if self._is_running:
                self.run_inspection()

        self.gpio.set_trigger_callback(on_trigger)
        self.gpio.start_trigger_monitor()

        logger.info("Continuous inspection mode started")

    def stop_continuous_mode(self):
        """Stop continuous inspection mode"""
        self._is_running = False
        self.gpio.stop_trigger_monitor()
        logger.info("Continuous inspection mode stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current inspection statistics"""
        total = self._ok_count + self._nok_count + self._error_count
        return {
            'total_inspections': self._inspection_count,
            'ok_count': self._ok_count,
            'nok_count': self._nok_count,
            'error_count': self._error_count,
            'ok_rate': (self._ok_count / total * 100) if total > 0 else 0,
            'nok_rate': (self._nok_count / total * 100) if total > 0 else 0,
            'error_rate': (self._error_count / total * 100) if total > 0 else 0
        }

    def cleanup(self):
        """Clean up all resources"""
        self.stop_continuous_mode()
        self.gpio.cleanup()
        self.camera.cleanup()
        self.ftp.cleanup()
        logger.info("Quality control system shut down")


def main():
    """Test entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='QC Hardware Automation')
    parser.add_argument('--test-gpio', action='store_true', help='Test GPIO relays')
    parser.add_argument('--test-camera', action='store_true', help='Test camera capture')
    parser.add_argument('--test-ftp', action='store_true', help='Test FTP upload')
    parser.add_argument('--run', action='store_true', help='Run continuous mode')

    args = parser.parse_args()

    if args.test_gpio:
        gpio = GPIOController()
        gpio.initialize()
        print("Testing relays...")
        gpio.signal_ok()
        time.sleep(1)
        gpio.signal_nok()
        time.sleep(1)
        gpio.set_light(True)
        time.sleep(2)
        gpio.set_light(False)
        gpio.cleanup()

    elif args.test_camera:
        camera = CameraController()
        camera.initialize()
        path = camera.capture('test_capture.jpg')
        print(f"Captured: {path}")
        camera.cleanup()

    elif args.test_ftp:
        ftp = FTPUploader()
        # Create test file
        test_file = '/tmp/ftp_test.txt'
        with open(test_file, 'w') as f:
            f.write(f"Test upload at {datetime.now()}")
        result = ftp.upload(test_file)
        print(f"Upload result: {result}")
        ftp.cleanup()

    elif args.run:
        system = QualityControlSystem()
        if system.initialize():
            system.start_continuous_mode()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            system.cleanup()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
