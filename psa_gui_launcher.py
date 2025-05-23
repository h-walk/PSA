#!/usr/bin/env python3
"""
PSA GUI Launcher

Simple launcher script for the PSA GUI application.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from psa.gui.psa_gui import main
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the PSA package is properly installed.")
    print("Try running: pip install -e .")
except Exception as e:
    print(f"Error starting GUI: {e}")
    sys.exit(1) 