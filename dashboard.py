#!/usr/bin/env python3
"""
Real-time Dashboard for Quality Control System

Provides a web-based interface for operators to:
- View live inspection statistics
- Receive alert notifications for detected anomalies
- Review and classify past detections
- Monitor system health and status

Uses Flask with Socket.IO for real-time updates.
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional

from flask import Flask, render_template, jsonify, request, send_file, abort
from flask_socketio import SocketIO, emit

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qc_main import get_daemon, QualityControlDaemon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'qc_system_secret_key_change_in_production'

# Socket.IO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global reference to daemon
daemon: Optional[QualityControlDaemon] = None


def init_daemon():
    """Initialize or get the daemon instance"""
    global daemon
    if daemon is None:
        daemon = get_daemon()
        if not daemon.state.session_start_time:
            daemon.initialize()
        # Register event callback for real-time updates
        daemon.add_event_callback(handle_daemon_event)
    return daemon


def handle_daemon_event(event: Dict[str, Any]):
    """Handle events from the QC daemon and broadcast to clients"""
    event_type = event.get('type')
    data = event.get('data', {})

    if event_type == 'inspection_complete':
        socketio.emit('inspection_result', data)
        # Also send updated statistics
        stats = daemon.get_statistics()
        socketio.emit('statistics_update', stats)

    elif event_type == 'anomaly_detected':
        socketio.emit('anomaly_alert', data)

    elif event_type == 'system_started':
        socketio.emit('system_status', {'status': 'running'})

    elif event_type == 'system_stopped':
        socketio.emit('system_status', {'status': 'stopped'})

    elif event_type == 'model_trained':
        socketio.emit('model_update', data)


# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current system status"""
    d = init_daemon()
    return jsonify({
        'success': True,
        'status': {
            'is_running': d.state.is_running,
            'model_loaded': d.state.model_loaded,
            'session_start': d.state.session_start_time,
            'last_inspection': d.state.last_inspection_time
        }
    })


@app.route('/api/statistics')
def get_statistics():
    """Get system statistics"""
    d = init_daemon()
    return jsonify({
        'success': True,
        'statistics': d.get_statistics()
    })


@app.route('/api/results')
def get_results():
    """Get recent inspection results"""
    d = init_daemon()
    count = request.args.get('count', 50, type=int)
    return jsonify({
        'success': True,
        'results': d.get_recent_results(count)
    })


@app.route('/api/anomalies')
def get_anomalies():
    """Get recent anomaly detections"""
    d = init_daemon()
    count = request.args.get('count', 20, type=int)
    return jsonify({
        'success': True,
        'anomalies': d.get_anomalies(count)
    })


@app.route('/api/result/<inspection_id>')
def get_result_detail(inspection_id):
    """Get detailed result for specific inspection"""
    d = init_daemon()
    result = d.result_store.get_by_id(inspection_id)
    if result:
        return jsonify({
            'success': True,
            'result': result
        })
    return jsonify({
        'success': False,
        'error': 'Result not found'
    }), 404


@app.route('/api/image/<path:image_path>')
def get_image(image_path):
    """Serve inspection images"""
    # Security: only serve from allowed directories
    allowed_dirs = ['/tmp/qc_captures', '/var/lib/qc_system']
    full_path = '/' + image_path

    is_allowed = any(full_path.startswith(d) for d in allowed_dirs)
    if not is_allowed or not os.path.exists(full_path):
        abort(404)

    return send_file(full_path, mimetype='image/jpeg')


@app.route('/api/control/start', methods=['POST'])
def start_system():
    """Start the quality control system"""
    d = init_daemon()
    d.start()
    return jsonify({
        'success': True,
        'message': 'System started'
    })


@app.route('/api/control/stop', methods=['POST'])
def stop_system():
    """Stop the quality control system"""
    d = init_daemon()
    d.stop()
    return jsonify({
        'success': True,
        'message': 'System stopped'
    })


@app.route('/api/control/trigger', methods=['POST'])
def trigger_inspection():
    """Trigger a manual inspection"""
    d = init_daemon()
    result = d.trigger_manual_inspection()
    return jsonify({
        'success': True,
        'result': result
    })


@app.route('/api/classify', methods=['POST'])
def classify_result():
    """Operator classification of an inspection result"""
    d = init_daemon()
    data = request.json
    inspection_id = data.get('inspection_id')
    classification = data.get('classification')

    if not inspection_id or not classification:
        return jsonify({
            'success': False,
            'error': 'Missing inspection_id or classification'
        }), 400

    success = d.classify_result(inspection_id, classification)
    return jsonify({
        'success': success,
        'message': 'Classification saved' if success else 'Failed to save'
    })


@app.route('/api/train', methods=['POST'])
def train_model():
    """Start model training"""
    d = init_daemon()
    data = request.json
    training_dir = data.get('training_dir')

    if not training_dir or not os.path.isdir(training_dir):
        return jsonify({
            'success': False,
            'error': 'Invalid training directory'
        }), 400

    # Run training in background
    def train_thread():
        def progress(current, total):
            socketio.emit('training_progress', {
                'current': current,
                'total': total,
                'percent': int(current * 100 / total)
            })

        success = d.train_model(training_dir, progress)
        socketio.emit('training_complete', {
            'success': success
        })

    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()

    return jsonify({
        'success': True,
        'message': 'Training started'
    })


# ============================================================================
# Socket.IO Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    d = init_daemon()
    # Send current status to new client
    emit('system_status', {
        'status': 'running' if d.state.is_running else 'stopped',
        'model_loaded': d.state.model_loaded
    })
    # Send current statistics
    emit('statistics_update', d.get_statistics())


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('request_statistics')
def handle_request_statistics():
    """Handle statistics request from client"""
    d = init_daemon()
    emit('statistics_update', d.get_statistics())


@socketio.on('request_results')
def handle_request_results(data):
    """Handle results request from client"""
    d = init_daemon()
    count = data.get('count', 50)
    emit('results_update', {
        'results': d.get_recent_results(count)
    })


@socketio.on('request_anomalies')
def handle_request_anomalies(data):
    """Handle anomalies request from client"""
    d = init_daemon()
    count = data.get('count', 20)
    emit('anomalies_update', {
        'anomalies': d.get_anomalies(count)
    })


# ============================================================================
# Main Entry Point
# ============================================================================

def create_app():
    """Create and configure the Flask application"""
    # Create template directory if needed
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    return app


def main():
    """Main entry point for dashboard server"""
    import argparse

    parser = argparse.ArgumentParser(description='QC Dashboard Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    logger.info(f"Starting dashboard server on {args.host}:{args.port}")

    # Initialize daemon
    init_daemon()

    # Run server
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False  # Disable reloader in production
    )


if __name__ == '__main__':
    main()
