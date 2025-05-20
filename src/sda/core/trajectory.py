"""
Core trajectory data structure for molecular dynamics data.
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class Trajectory:
    positions: np.ndarray
    velocities: np.ndarray
    types: np.ndarray
    timesteps: np.ndarray
    box_matrix: np.ndarray
    box_lengths: np.ndarray
    box_tilts: np.ndarray
    dt_ps: float # Timestep in picoseconds

    def __post_init__(self):
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError("Positions must be 3D (frames, atoms, xyz) and last dimension must be 3.")
        if self.velocities.ndim != 3 or self.velocities.shape[2] != 3:
            raise ValueError("Velocities must be 3D (frames, atoms, xyz) and last dimension must be 3.")
        if self.types.ndim != 1:
            raise ValueError("Types must be 1D")
        if self.timesteps.ndim != 1:
            raise ValueError("Timesteps must be 1D")
        if not (self.positions.shape[0] == self.velocities.shape[0] == len(self.timesteps)):
            raise ValueError("Frame count mismatch: positions, velocities, timesteps.")
        if not (self.positions.shape[1] == self.velocities.shape[1] == len(self.types)):
            raise ValueError("Atom count mismatch: positions, velocities, types.")
        if self.box_matrix.shape != (3, 3):
            raise ValueError(f"Box matrix must be 3x3, got {self.box_matrix.shape}")
        if self.box_lengths.shape != (3,):
            raise ValueError(f"Box lengths must be a 3-element array, got {self.box_lengths.shape}")
        if self.box_tilts.shape != (3,):
            raise ValueError(f"Box tilts must be a 3-element array, got {self.box_tilts.shape}")

    @property
    def n_frames(self) -> int:
        return len(self.timesteps)

    @property
    def n_atoms(self) -> int:
        return len(self.types) 