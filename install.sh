#!/bin/bash
# Installation script for AI Quality Control System
# Run as root on Raspberry Pi 5

set -e

echo "=============================================="
echo "AI Quality Control System Installation"
echo "=============================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./install.sh)"
    exit 1
fi

# Configuration
QC_USER="ai2"
QC_HOME="/home/${QC_USER}"
QC_DIR="${QC_HOME}/qc_system"
IPC_DIR="${QC_HOME}/yolo_ipc"
LOG_DIR="/var/log/qc_system"
DATA_DIR="/var/lib/qc_system"

echo ""
echo "Step 1: Creating user and directories..."
echo "----------------------------------------"

# Create user if doesn't exist
if ! id "${QC_USER}" &>/dev/null; then
    useradd -m -s /bin/bash "${QC_USER}"
    echo "Created user: ${QC_USER}"
fi

# Add user to gpio and video groups
usermod -aG gpio,video "${QC_USER}" 2>/dev/null || true

# Create directories
mkdir -p "${QC_DIR}"
mkdir -p "${IPC_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "/tmp/qc_captures"

# Set permissions
chown -R "${QC_USER}:${QC_USER}" "${QC_HOME}"
chown -R "${QC_USER}:${QC_USER}" "${LOG_DIR}"
chown -R "${QC_USER}:${QC_USER}" "${DATA_DIR}"
chown -R "${QC_USER}:${QC_USER}" "/tmp/qc_captures"

echo ""
echo "Step 2: Installing system dependencies..."
echo "-----------------------------------------"

apt-get update
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-gpiod \
    python3-picamera2 \
    libcamera-apps \
    libcamera-dev

echo ""
echo "Step 3: Copying application files..."
echo "------------------------------------"

# Copy application files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "${SCRIPT_DIR}/dariusz.py" "${QC_DIR}/"
cp "${SCRIPT_DIR}/automation.py" "${QC_DIR}/"
cp "${SCRIPT_DIR}/qc_main.py" "${QC_DIR}/"
cp "${SCRIPT_DIR}/dashboard.py" "${QC_DIR}/"
cp "${SCRIPT_DIR}/requirements.txt" "${QC_DIR}/"

# Copy templates and static files
mkdir -p "${QC_DIR}/templates"
mkdir -p "${QC_DIR}/static"
cp -r "${SCRIPT_DIR}/templates/"* "${QC_DIR}/templates/" 2>/dev/null || true
cp -r "${SCRIPT_DIR}/static/"* "${QC_DIR}/static/" 2>/dev/null || true

chown -R "${QC_USER}:${QC_USER}" "${QC_DIR}"

echo ""
echo "Step 4: Installing Python dependencies..."
echo "-----------------------------------------"

# Create virtual environment
cd "${QC_DIR}"
sudo -u "${QC_USER}" python3 -m venv venv
sudo -u "${QC_USER}" "${QC_DIR}/venv/bin/pip" install --upgrade pip
sudo -u "${QC_USER}" "${QC_DIR}/venv/bin/pip" install -r requirements.txt

echo ""
echo "Step 5: Installing systemd services..."
echo "--------------------------------------"

# Copy systemd service files
cp "${SCRIPT_DIR}/systemd/yolo_ai_daemon.service" /etc/systemd/system/
cp "${SCRIPT_DIR}/systemd/qc_dashboard.service" /etc/systemd/system/

# Update paths in service files to use virtual environment
sed -i "s|/usr/bin/python3|${QC_DIR}/venv/bin/python|g" /etc/systemd/system/yolo_ai_daemon.service
sed -i "s|/usr/bin/python3|${QC_DIR}/venv/bin/python|g" /etc/systemd/system/qc_dashboard.service
sed -i "s|/home/ai2/qc_system|${QC_DIR}|g" /etc/systemd/system/yolo_ai_daemon.service
sed -i "s|/home/ai2/qc_system|${QC_DIR}|g" /etc/systemd/system/qc_dashboard.service

# Reload systemd
systemctl daemon-reload

echo ""
echo "Step 6: Enabling services for auto-start..."
echo "--------------------------------------------"

systemctl enable yolo_ai_daemon.service
systemctl enable qc_dashboard.service

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Services installed:"
echo "  - yolo_ai_daemon.service (AI inference daemon)"
echo "  - qc_dashboard.service (Web dashboard)"
echo ""
echo "Directories:"
echo "  - Application: ${QC_DIR}"
echo "  - IPC: ${IPC_DIR}"
echo "  - Logs: ${LOG_DIR}"
echo "  - Data: ${DATA_DIR}"
echo ""
echo "To start the services:"
echo "  sudo systemctl start yolo_ai_daemon"
echo "  sudo systemctl start qc_dashboard"
echo ""
echo "To check status:"
echo "  sudo systemctl status yolo_ai_daemon"
echo "  sudo systemctl status qc_dashboard"
echo ""
echo "Dashboard URL: http://<raspberry-pi-ip>:5000"
echo ""
echo "To train the model, place good images in a folder and run:"
echo "  sudo -u ${QC_USER} ${QC_DIR}/venv/bin/python ${QC_DIR}/qc_main.py --train /path/to/good/images"
echo ""
echo "Note: Reboot the system to ensure all services start correctly."
