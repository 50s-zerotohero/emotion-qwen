"""Launch the Emotion Probe Gradio UI.

Usage:
    python scripts/05_launch_ui.py [--port PORT] [--share]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_probe.ui.app import launch

parser = argparse.ArgumentParser()
parser.add_argument("--port",  type=int, default=7860, help="Gradio server port")
parser.add_argument("--share", action="store_true",    help="Create public Gradio share link")
args = parser.parse_args()

launch(share=args.share, server_port=args.port)
