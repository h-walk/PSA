"""
Plot styling module for PSA.

This module provides predefined styles and color schemes for PSA plots.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any, Optional

# Default style parameters
DEFAULT_STYLE = {
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
    'figure.autolayout': True,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'image.cmap': 'viridis',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False
}

# Color schemes
COLOR_SCHEMES = {
    'default': {
        'primary': '#1f77b4',  # Blue
        'secondary': '#ff7f0e',  # Orange
        'tertiary': '#2ca02c',  # Green
        'quaternary': '#d62728',  # Red
        'background': '#ffffff',  # White
        'grid': '#cccccc'  # Light gray
    },
    'dark': {
        'primary': '#4c72b0',  # Light blue
        'secondary': '#dd8452',  # Light orange
        'tertiary': '#55a868',  # Light green
        'quaternary': '#c44e52',  # Light red
        'background': '#2d2d2d',  # Dark gray
        'grid': '#404040'  # Medium gray
    },
    'scientific': {
        'primary': '#000000',  # Black
        'secondary': '#e41a1c',  # Red
        'tertiary': '#377eb8',  # Blue
        'quaternary': '#4daf4a',  # Green
        'background': '#ffffff',  # White
        'grid': '#dddddd'  # Light gray
    }
}

def apply_style(style: Optional[Dict[str, Any]] = None, color_scheme: str = 'default') -> None:
    """
    Apply a style to matplotlib plots.
    
    Args:
        style: Dictionary of style parameters to override defaults
        color_scheme: Name of the color scheme to use ('default', 'dark', or 'scientific')
    """
    if style is None:
        style = {}
    
    # Get the selected color scheme
    if color_scheme not in COLOR_SCHEMES:
        raise ValueError(f"Unknown color scheme: {color_scheme}. Must be one of: {list(COLOR_SCHEMES.keys())}")
    colors = COLOR_SCHEMES[color_scheme]
    
    # Update style with color scheme
    style.update({
        'axes.facecolor': colors['background'],
        'figure.facecolor': colors['background'],
        'grid.color': colors['grid'],
        'axes.edgecolor': colors['primary'],
        'axes.labelcolor': colors['primary'],
        'xtick.color': colors['primary'],
        'ytick.color': colors['primary'],
        'text.color': colors['primary']
    })
    
    # Apply the style
    plt.style.use(style)
    
def get_colormap(name: str = 'viridis') -> mpl.colors.Colormap:
    """
    Get a colormap by name.
    
    Args:
        name: Name of the colormap
        
    Returns:
        Matplotlib colormap object
    """
    return plt.get_cmap(name)

def get_color_cycle() -> list:
    """
    Get the current color cycle.
    
    Returns:
        List of colors in the current cycle
    """
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def set_color_cycle(colors: list) -> None:
    """
    Set the color cycle for plots.
    
    Args:
        colors: List of colors to use in the cycle
    """
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def get_style_params() -> Dict[str, Any]:
    """
    Get the current style parameters.
    
    Returns:
        Dictionary of current style parameters
    """
    return {k: v for k, v in plt.rcParams.items() if k in DEFAULT_STYLE}

def reset_style() -> None:
    """Reset the style to matplotlib defaults."""
    plt.style.use('default') 