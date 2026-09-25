#!/usr/bin/env python3
"""
AstroHAR — Astronaut Hand Activity Recognition
Mission Control Real-Time Dashboard Server Entrypoint
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.har_space.dashboard.server import start_dashboard_server

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
        
    try:
        start_dashboard_server(port=port)
    except KeyboardInterrupt:
        print("\n[AstroHAR Dashboard] Server shutdown cleanly.")
