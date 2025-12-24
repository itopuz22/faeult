# AI Quality Control System

Industrial-grade quality control system for manufacturing using PatchCore anomaly detection on Raspberry Pi 5 with AI Hat.

## Features

- **PatchCore Anomaly Detection**: WideResNet50 backbone with coreset sampling for efficient memory bank
- **Defect Classification**: Detects damage, scratches, and dents with confidence scores
- **Real-time Dashboard**: Live statistics, alerts, and detection review interface
- **Fault Tolerance**: Automatic recovery after power failures via systemd
- **Hardware Integration**: GPIO relay control, HQ camera, FTP upload

## Hardware Requirements

- Raspberry Pi 5
- AI Hat (for ML inference acceleration)
- Relay board with K1 (OK), K2 (Light), K3 (NOK) outputs
- Raspberry HQ Camera
- GPIO connections as per pin configuration

## GPIO Pin Configuration

| Function | GPIO Pin | Description |
|----------|----------|-------------|
| Input Trigger | GPIO 17 | Trigger signal from production line |
| K1 OK | GPIO 26 | Relay output for good parts |
| K2 Light | GPIO 20 | Lighting control relay |
| K3 NOK | GPIO 21 | Relay output for defective parts |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd qc-system

# Run installation script (as root)
sudo ./install.sh
```

## Usage

### Starting Services

```bash
# Start AI daemon
sudo systemctl start yolo_ai_daemon

# Start dashboard
sudo systemctl start qc_dashboard

# Check status
sudo systemctl status yolo_ai_daemon
sudo systemctl status qc_dashboard
```

### Training the Model

Before use, train the model on images of good (non-defective) parts:

```bash
# Place good images in a folder
sudo -u ai2 /home/ai2/qc_system/venv/bin/python \
    /home/ai2/qc_system/qc_main.py --train /path/to/good/images
```

### Dashboard Access

Open a web browser and navigate to:
```
http://<raspberry-pi-ip>:5000
```

## Architecture

```
├── dariusz.py          # PatchCore anomaly detection model
├── automation.py       # Hardware control (GPIO, camera, FTP)
├── qc_main.py          # Main application daemon
├── dashboard.py        # Flask web dashboard
├── templates/          # HTML templates
├── static/             # Static assets
└── systemd/            # Service definitions
```

## IPC Protocol

Communication between components uses JSON files in `/home/ai2/yolo_ipc/`:

**Request format:**
```json
{
    "command": "infer",
    "image_path": "/tmp/qc_captures/image.jpg",
    "timestamp": "2024-01-15T10:30:00"
}
```

**Response format:**
```json
{
    "status": "success",
    "result": {
        "is_anomaly": true,
        "defect_type": "scratch",
        "confidence": 0.92,
        "anomaly_score": 1.23
    }
}
```

## FTP Configuration

Images are uploaded to the configured FTP server:
- Host: 10.5.191.182
- Path: /A10_AI/{DDMMYY}/OK or /A10_AI/{DDMMYY}/NOK

## Systemd Services

### yolo_ai_daemon.service
Main quality control daemon with:
- Automatic restart on failure
- 5-second restart delay
- Watchdog monitoring

### qc_dashboard.service
Web dashboard server:
- Depends on yolo_ai_daemon
- Automatic restart
- Binds to port 5000

## State Persistence

System state is preserved across restarts in `/var/lib/qc_system/`:
- `system_state.json`: Running state and statistics
- `inspection_results.json`: Recent inspection results
- `patchcore_model.pkl`: Trained anomaly detection model

## Logs

View logs using journalctl:
```bash
# AI daemon logs
journalctl -u yolo_ai_daemon -f

# Dashboard logs
journalctl -u qc_dashboard -f

# Application logs
tail -f /var/log/qc_system/qc_system.log
```

## License

Proprietary - Internal Use Only
