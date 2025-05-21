"""
Visualization module for PSA.

This module provides plotting capabilities for SED data.
"""

from .sed_plotter import SEDPlotter
from .styles import apply_style, DEFAULT_STYLE, COLOR_SCHEMES

__all__ = [
    'SEDPlotter',
    'apply_style',
    'DEFAULT_STYLE',
    'COLOR_SCHEMES'
]
