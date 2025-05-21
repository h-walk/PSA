"""
Core spectral energy density (SED) data structure.
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SED:
    sed: np.ndarray
    freqs: np.ndarray
    k_points: np.ndarray  # Magnitudes of k-vectors (e.g., for 1D path plots) or other 1D representation if k_vectors is a grid.
    k_vectors: np.ndarray  # Full 3D k-vectors, shape (n_k_points, 3)
    k_grid_shape: Optional[Tuple[int, ...]] = None # Shape of the k-point grid, e.g., (nkx, nky) for a 2D grid. None for a k-path.
    phase: Optional[np.ndarray] = None
    is_complex: bool = True  # Indicates if sed attribute holds complex amplitudes or intensities

    @property
    def intensity(self) -> np.ndarray:
        return np.sum(np.abs(self.sed)**2, axis=-1).astype(np.float32)

    def save(self, base_path: Path):
        base_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(base_path.with_suffix('.sed.npy'), self.sed)
        np.save(base_path.with_suffix('.freqs.npy'), self.freqs)
        np.save(base_path.with_suffix('.k_points.npy'), self.k_points)
        np.save(base_path.with_suffix('.k_vectors.npy'), self.k_vectors)
        if self.k_grid_shape is not None:
            # Save k_grid_shape as a simple text file or a .npy file
            # Using .npy for consistency, though it's a small tuple
            np.save(base_path.with_suffix('.k_grid_shape.npy'), np.array(self.k_grid_shape))
        if self.phase is not None:
            np.save(base_path.with_suffix('.phase.npy'), self.phase)
        logger.info(f"SED data saved: {base_path.name}.*.npy")

    @staticmethod
    def load(base_path: Path) -> 'SED':
        required_suffixes = ['.sed.npy', '.freqs.npy', '.k_points.npy', '.k_vectors.npy']
        if not all((base_path.with_suffix(s)).exists() for s in required_suffixes):
            raise FileNotFoundError(f"Required SED files missing for base: {base_path.name}")

        sed_val = np.load(base_path.with_suffix('.sed.npy'))
        freqs_val = np.load(base_path.with_suffix('.freqs.npy'))
        k_points_val = np.load(base_path.with_suffix('.k_points.npy'))
        k_vectors_val = np.load(base_path.with_suffix('.k_vectors.npy'))
        phase_val = None
        phase_file = base_path.with_suffix('.phase.npy')
        if phase_file.exists():
            try:
                phase_val = np.load(phase_file)
            except Exception as e:
                logger.warning(f"Could not load phase data from {phase_file.name}: {e}")
        
        k_grid_shape_val = None
        k_grid_shape_file = base_path.with_suffix('.k_grid_shape.npy')
        if k_grid_shape_file.exists():
            try:
                k_grid_shape_val_loaded = np.load(k_grid_shape_file)
                # Ensure it's converted to a tuple of ints
                k_grid_shape_val = tuple(map(int, k_grid_shape_val_loaded))
            except Exception as e:
                logger.warning(f"Could not load k_grid_shape data from {k_grid_shape_file.name}: {e}")

        return SED(sed_val, freqs_val, k_points_val, k_vectors_val, 
                   k_grid_shape=k_grid_shape_val, phase=phase_val) 