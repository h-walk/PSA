"""
Utility functions for PSA.

This module provides helper functions for various PSA operations.
"""
import numpy as np
import logging
from typing import Union, List, Tuple, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_direction(direction_spec: Union[str, int, float, List[float], Tuple[float, ...], np.ndarray, Dict[str, float]]) -> np.ndarray:
    """
    Parse a direction specification into a normalized 3D vector.
    
    Args:
        direction_spec: Direction specification, which can be:
            - String: 'x', 'y', 'z', 'xy', 'yz', 'xz', 'xyz', or angle in degrees
            - Number: Angle in degrees (in XY plane)
            - List/Tuple/Array: [x,y,z] components
            - Dict: {'angle': deg} or {'h': h, 'k': k, 'l': l} for Miller indices
            
    Returns:
        Normalized 3D direction vector
        
    Raises:
        ValueError: If direction specification is invalid
        TypeError: If direction specification type is not supported
    """
    vec = np.zeros(3, dtype=np.float32)
    
    if isinstance(direction_spec, (int, float)): 
        rad = np.deg2rad(float(direction_spec))
        vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        
    elif isinstance(direction_spec, str):
        d_lower = direction_spec.lower()
        mapping = {
            'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1],
            'xy': [1/np.sqrt(2),1/np.sqrt(2),0],
            'yx': [1/np.sqrt(2),1/np.sqrt(2),0],
            'xz': [1/np.sqrt(2),0,1/np.sqrt(2)],
            'zx': [1/np.sqrt(2),0,1/np.sqrt(2)],
            'yz': [0,1/np.sqrt(2),1/np.sqrt(2)],
            'zy': [0,1/np.sqrt(2),1/np.sqrt(2)],
            'xyz': [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
            '100': [1,0,0], '010': [0,1,0], '001': [0,0,1],
            '110': [1/np.sqrt(2),1/np.sqrt(2),0],
            '111': [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]
        }
        if d_lower in mapping:
            vec = np.array(mapping[d_lower], dtype=np.float32)
        else:
            try: # Try parsing as angle AFTER checking mapping
                angle_deg = float(direction_spec)
                rad = np.deg2rad(angle_deg)
                vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
            except ValueError: # If not in mapping and not an angle, try parsing as x,y,z components
                try:
                    parts = direction_spec.replace(',', ' ').split()
                    if len(parts) == 3:
                        vec = np.array([float(p) for p in parts], dtype=np.float32)
                    else:
                        raise ValueError() # Will be caught by outer except
                except ValueError:
                    raise ValueError(f"Unknown direction string: {direction_spec}.")
                    
    elif isinstance(direction_spec, (list, tuple, np.ndarray)):
        d_arr = np.asarray(direction_spec, dtype=np.float32).squeeze() 
        if d_arr.ndim == 0: 
            rad = np.deg2rad(d_arr.item())
            vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        elif d_arr.ndim == 1:
            if d_arr.size == 1:
                rad = np.deg2rad(d_arr[0])
                vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
            elif d_arr.size == 3:
                vec = d_arr
            else:
                raise ValueError(f"Direction array must have 1 (angle) or 3 (vector) components, got {d_arr.size}")
        else:
            raise ValueError(f"Direction array has too many dims: {d_arr.ndim}, expected 0 or 1 (squeezed).")
            
    elif isinstance(direction_spec, dict): 
        if 'angle' in direction_spec: 
            rad = np.deg2rad(float(direction_spec['angle']))
            vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        elif any(k in direction_spec for k in ['h', 'k', 'l']): 
            vec = np.array([
                float(direction_spec.get('h',0.0)),
                float(direction_spec.get('k',0.0)),
                float(direction_spec.get('l',0.0))
            ], dtype=np.float32)
        else:
            raise ValueError("Direction dict must contain 'angle' or Miller indices ('h','k','l').")
    else:
        raise TypeError(f"Unsupported direction type: {type(direction_spec)}")

    # Check if the vector is effectively zero first
    if np.allclose(vec, 0, atol=1e-8): # Using atol consistent with np.allclose default
        raise ValueError("Direction vector is zero. For k-path, direction must be non-zero if n_k > 1.")

    norm_val = np.linalg.norm(vec)
    if norm_val < 1e-9: # This threshold is for *very* small norms, but not necessarily "allclose" to zero
        logger.warning(f"Direction vector norm ({norm_val:.2e}) is very small, returning unnormalized vector.")
        return vec
            
    return vec / norm_val

def update_dict_recursively(base_dict: dict, update_with: dict) -> dict:
    """
    Recursively update a dictionary with another dictionary.
    
    Args:
        base_dict: Base dictionary to update
        update_with: Dictionary containing updates
        
    Returns:
        Updated dictionary
    """
    for k, v_update in update_with.items():
        if isinstance(v_update, dict) and k in base_dict and isinstance(base_dict[k], dict):
            update_dict_recursively(base_dict[k], v_update)
        else:
            base_dict[k] = v_update
    return base_dict

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def validate_array_shape(arr: np.ndarray, expected_shape: tuple, name: str) -> None:
    """
    Validate that an array has the expected shape.
    
    Args:
        arr: Array to validate
        expected_shape: Expected shape tuple
        name: Name of the array for error messages
        
    Raises:
        ValueError: If array shape doesn't match expected shape
    """
    if arr.shape != expected_shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected_shape}")

def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Args:
        a: Numerator array
        b: Denominator array
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result of division, with fill_value where denominator is zero
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b, out=np.full_like(a, fill_value), where=b!=0)
    return result 