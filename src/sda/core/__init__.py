"""
Core module for SDA.

This module provides the core data structures and calculation engines for SDA.
"""

from .trajectory import Trajectory
from .sed import SED
from .sed_calculator import SEDCalculator
# SEDPlotter is in the visualization module, not core.
# We will adjust the test imports later if SEDPlotter was indeed moved.

__all__ = [
    'Trajectory',
    'SED',
    'SEDCalculator',
]
