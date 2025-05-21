"""
Spectral Displacement Analysis (SDA) Package
"""

__version__ = "0.1.0" # Consider linking this to setup.py version

# Core components
from .core.trajectory import Trajectory
from .core.sed import SED
from .core.sed_calculator import SEDCalculator

# IO components
from .io.loader import TrajectoryLoader
from .io.writer import TrajectoryWriter, out_to_qdump

# Visualization components
from .visualization import SEDPlotter
from .visualization.styles import apply_style, DEFAULT_STYLE, COLOR_SCHEMES

# Utility components
from .utils.helpers import (
    parse_direction,
    update_dict_recursively,
    ensure_directory,
    validate_array_shape,
    safe_divide
)
from .utils.config_manager import ConfigManager # If it's intended for public API

# Main CLI function (optional, if you want to allow programmatic execution of CLI)
# from .cli import main as run_sda_cli

__all__ = [
    # Core
    'Trajectory',
    'SED',
    'SEDCalculator',
    # IO
    'TrajectoryLoader',
    'TrajectoryWriter',
    'out_to_qdump',
    # Visualization
    'SEDPlotter',
    'apply_style',
    'DEFAULT_STYLE',
    'COLOR_SCHEMES',
    # Utils
    'parse_direction',
    'update_dict_recursively',
    'ensure_directory',
    'validate_array_shape',
    'safe_divide',
    'ConfigManager',
    # 'run_sda_cli' # Optional CLI exposure
]
