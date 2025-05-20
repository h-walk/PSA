# SDA API Reference

This section provides detailed API documentation for the `sda` package.

## Modules Overview

The `sda` package is organized into the following main modules:

- **`sda.core`**: Core data structures and the main calculation engine.
  - [`Trajectory`](core/trajectory.md): Data structure for molecular dynamics trajectories.
  - [`SED`](core/sed.md): Data structure for Spectral Energy Density and related data.
  - [`SEDCalculator`](core/sed_calculator.md): Main class for performing SED and iSED calculations.
- **`sda.io`**: Modules for data input/output.
  - [`TrajectoryLoader`](io/loader.md): Handles loading of trajectory data from files.
  - [`TrajectoryWriter`](io/writer.md): For writing trajectory data (if implemented for general use).
  - [`out_to_qdump`](io/writer.md): Function to write iSED reconstruction data to Qdump format.
- **`sda.visualization`**: Plotting and visualization tools.
  - [`SEDPlotter`](visualization/plotter.md): Class for generating various SED plots.
  - [`styles`](visualization/styles.md): Predefined plot styles and color schemes.
- **`sda.utils`**: Helper utilities.
  - [`helpers`](utils/helpers.md): Various utility functions (e.g., `parse_direction`, `ensure_directory`).
- **`sda.cli`**: Command-line interface.
  - While primarily for command-line use, it orchestrates the package's functionalities.

## Core Data Structures

### `sda.core.Trajectory`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Trajectory:
    positions: np.ndarray  # Shape: (frames, atoms, xyz)
    velocities: np.ndarray  # Shape: (frames, atoms, xyz)
    types: np.ndarray      # Shape: (atoms,)
    timesteps: np.ndarray  # Shape: (frames,)
    box_matrix: np.ndarray # Shape: (3, 3), full simulation box matrix
    box_lengths: np.ndarray # Shape: (3,), typically [Lx, Ly, Lz] for orthogonal
    box_tilts: np.ndarray  # Shape: (3,), typically [xy, xz, yz] for triclinic
```

### `sda.core.SED`

```python
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class SED:
    sed: np.ndarray        # Complex SED data, shape: (n_freqs, n_k_points, 3_polarizations)
    freqs: np.ndarray      # Frequencies (THz), shape: (n_freqs,)
    k_points: np.ndarray   # k-point magnitudes (2π/Å), shape: (n_k_points,)
    k_vectors: np.ndarray  # Actual 3D k-vectors (2π/Å), shape: (n_k_points, 3)
    phase: Optional[np.ndarray] = None  # Phase data for chiral analysis, shape: (n_freqs, n_k_points)
```

## Main Classes

### `sda.core.SEDCalculator`

The primary class for performing SED and iSED calculations.

```python
from .trajectory import Trajectory
from .sed import SED
from ..utils.helpers import parse_direction # Relative import as it would be in the module
from ..io.writer import out_to_qdump       # Relative import
from ..visualization import SEDPlotter   # Relative import
from typing import Tuple, List, Optional, Union, Dict
import numpy as np
from pathlib import Path

class SEDCalculator:
    def __init__(self, traj: Trajectory, nx: int, ny: int, nz: int, dt_ps: float, use_velocities: bool = False):
        """
        Initialize SED calculator.
        
        Args:
            traj: Trajectory object.
            nx, ny, nz: Number of primitive cells the simulation box corresponds to in x, y, z.
            dt_ps: Timestep of the MD simulation in picoseconds.
            use_velocities: If True, use velocities for SED calculation. Otherwise, use displacements.
        """
        pass

    def get_k_path(self, 
                   direction_spec: Union[str, int, float, List[float], Tuple[float, ...], np.ndarray, Dict[str, float]],
                   bz_coverage: float, 
                   n_k: int, 
                   lat_param: Optional[float] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a k-path (magnitudes and 3D vectors) for SED calculation.
        
        Args:
            direction_spec: Specification for the k-path direction.
                            Examples: "x", [1,1,0], {"angle": 45}, {"h":1, "k":0, "l":0}.
            bz_coverage: How far to go along the direction, typically in units of π/lat_param.
                         E.g., 1.0 means from Gamma to the BZ boundary in that direction.
            n_k: Number of k-points along the path.
            lat_param: Optional characteristic lattice parameter (e.g., Angstroms) to define k-path length.
                       If None, |a1| (primitive vector norm) from the calculator is used.
            
        Returns:
            Tuple of (k_magnitudes, k_vectors_3d).
            k_magnitudes: 1D array of k-point magnitudes (2π/Å).
            k_vectors_3d: 2D array of 3D k-vectors (2π/Å), shape (n_k, 3).
        """
        pass

    def calculate(self, 
                  k_points_mags: np.ndarray, 
                  k_vectors_3d: np.ndarray, 
                  basis_atom_indices: Optional[Union[List[int], np.ndarray]] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SED (displacement or velocity spectra) for the given k-points.
        
        Args:
            k_points_mags: 1D array of k-point magnitudes (not directly used in FFT, but for reference).
            k_vectors_3d: 2D array of actual 3D k-vectors used in the calculation.
            basis_atom_indices: Optional list or array of 0-based atom indices to include in the calculation.
                                If None, all atoms are used.
            
        Returns:
            Tuple of (sed_complex_wk, freqs_thz).
            sed_complex_wk: Complex SED data array, shape (n_frequencies, n_k_points, 3_polarizations).
            freqs_thz: 1D array of frequencies in THz.
        """
        pass

    def calculate_chiral_phase(self, Z1: np.ndarray, Z2: np.ndarray, angle_range_opt: str = "C") -> np.ndarray:
        """
        Calculate the phase difference between two complex signals Z1 and Z2.
        Typically used for chiral SED from two polarization components.

        Args:
            Z1: Complex data array (e.g., SED for polarization 1). Shape (n_freqs, n_k_points).
            Z2: Complex data array (e.g., SED for polarization 2). Shape (n_freqs, n_k_points).
            angle_range_opt: Option for phase calculation method and range.
                             "C": Default, robust method returning phase in [-π/2, π/2].

        Returns:
            Phase difference array in radians.
        """
        pass
    
    def ised(self, k_dir_spec: Union[str, int, float, List[float], np.ndarray, Dict[str,float]],
             k_target: float, w_target: float, char_len_k_path: float,
             nk_on_path: int = 100, bz_cov_ised: float = 1.0,
             basis_atom_idx_ised: Optional[List[int]] = None, 
             basis_atom_types_ised: Optional[List[int]] = None,
             rescale_factor: Union[str, float] = 1.0, n_recon_frames: int = 100,
             dump_filepath: str = "iSED_reconstruction.dump",
             plot_dir_ised: Optional[Path] = None, plot_max_freq: Optional[float] = None
             ) -> None:
        """
        Perform Inverse Spectral Displacement (iSED) reconstruction for a specific mode.
        This involves calculating SED internally along a specified path to find the target mode,
        then reconstructing the atomic motion.

        Args:
            k_dir_spec: Direction for the k-path along which to search for the mode.
            k_target: Target k-point magnitude (2π/Å) for reconstruction.
            w_target: Target frequency (THz) for reconstruction.
            char_len_k_path: Characteristic length (e.g., lattice parameter in Å) for the k-path.
            nk_on_path: Number of k-points for the internal SED calculation.
            bz_cov_ised: Brillouin zone coverage for the internal SED.
            basis_atom_idx_ised: Atom indices for iSED (specific atoms or groups).
            basis_atom_types_ised: Atom types for iSED (used if indices not given).
            rescale_factor: Factor to rescale reconstructed amplitudes ('auto' or float).
            n_recon_frames: Number of frames for the output animation dump.
            dump_filepath: Path to save the LAMMPS dump file of reconstructed motion.
            plot_dir_ised: Directory to save a plot of the summed input SED spectrum for iSED.
            plot_max_freq: Max frequency for the iSED input spectrum plot.
        """
        pass
```

### `sda.visualization.SEDPlotter`

Class for generating plots from `SED` objects.

```python
from ..core.sed import SED # Relative import
from typing import Union, List
from pathlib import Path

class SEDPlotter:
    def __init__(self, sed_obj_or_list: Union[SED, List[SED]], plot_type: str, out_path_str: str, **kwargs):
        """
        Initialize SED plotter.
        
        Args:
            sed_obj_or_list: A single SED object (for 2D plots) or a list of SED objects (for 3D plots).
            plot_type: Type of plot. Supported: '2d_intensity', '2d_phase', 
                       '3d_intensity', '3d_phase'.
            out_path_str: Output path for the generated plot image file.
            **kwargs: Additional plotting parameters (e.g., title, cmap, max_freq, highlight_region).
                      See SEDPlotter implementation for all options.
        """
        pass

    def generate_plot(self):
        """Generate and save the plot to `out_path_str`."""
        pass
```

## CLI Configuration Example

The `sda` command-line tool uses a YAML configuration file. Below is an example structure:

```yaml
general:
  trajectory_file_format: 'auto' # 'lammps', 'vasp_outcar', or 'auto'
  use_velocities: False
  save_npy_trajectory: True
  save_npy_sed_data: True
  chiral_mode_enabled: False

md_system:
  dt: 0.001  # Timestep in picoseconds
  nx: 1      # Number of primitive cells in x direction of supercell
  ny: 1      # Number of primitive cells in y direction of supercell
  nz: 1      # Number of primitive cells in z direction of supercell
  lattice_parameter: null # Optional: characteristic length (e.g. Angstroms) for k-path scaling. If null, |a1| is used.

sed_calculation:
  directions: # List of k-path directions
    - [1,0,0] # Example: Gamma-X
    - 'y'     # Example: Gamma-Y (string shorthand)
    # - {'angle': 45} # Example: 45 degrees in xy-plane
    # - {'h':1, 'k':1, 'l':0} # Miller indices
  n_kpoints: 100
  bz_coverage: 1.0 # How far to go along the k-path direction (in units of pi/lattice_parameter)
  polarization_indices_chiral: [0,1] # For chiral SED: indices of polarizations to use (e.g., x and y)
  basis:
    atom_indices: null # Optional: list of specific atom indices [0, 1, 2, ...]
                       # Can be a list of lists for grouped basis in iSED.
    atom_types: null   # Optional: list of atom types [1, 2, ...] (used if atom_indices is null)
                       # Can be a list of lists for grouped basis in iSED.

plotting:
  max_freq_2d: null # Optional: clip max frequency in 2D plots (THz)
  highlight_2d_intensity: # Optional: highlight a region in 2D intensity plots
    k_min: null    # k-value (2π/Å)
    k_max: null    # k-value (2π/Å)
    w_min: null    # frequency (THz)
    w_max: null    # frequency (THz)
    # Alternatively, for a single point:
    # k_point_target: 1.57 
    # freq_point_target: 10.0
  enable_3d_dispersion_plot: False # Set to True to generate 3D plots if multiple directions are computed
  3d_plot_settings:
    intensity_log_scale: True
    intensity_thresh_rel: 0.01 # Relative threshold for culling points in 3D intensity plots

ised:
  apply: False # Set to true to run iSED
  k_path:
    direction: 'x'
    characteristic_length: null # If null, md_system.lattice_parameter is used
    n_points: 50
    bz_coverage: null # If null, sed_calculation.bz_coverage is used
  target_point:
    k_value: 1.57 # Example: pi/2 if lattice_parameter is 2.0 Angstroms (units of 2pi/Angstrom)
    w_value_thz: 10.0 # THz
  basis: # Which atoms to include in iSED mode reconstruction
    atom_indices: null # List of indices, or list of lists for groups
    atom_types: null   # List of types, or list of lists for groups (used if atom_indices is null)
  reconstruction:
    rescaling_factor: 'auto' # 'auto' or a float value
    num_animation_timesteps: 100
    output_dump_filename: 'ised_motion.dump'
    plot_dir_ised: null # Directory to save iSED input spectrum plot, if null, uses main output_dir
    plot_max_freq: null # Max freq for iSED input spectrum plot, if null, plotting.max_freq_2d is used
```

## Utility Functions (`sda.utils.helpers`)

A selection of publicly available helper functions:

### `parse_direction`
```python
def parse_direction(direction_spec: Union[str, int, float, List[float], Tuple[float, ...], np.ndarray, Dict[str, float]]) -> np.ndarray:
    """
    Parse a direction specification into a normalized 3D vector.
    
    Args:
        direction_spec: Direction specification (string like 'x', '100', '1,1,0'; 
                        number for angle in degrees; list/array for vector components; 
                        dict for {'angle': deg} or Miller indices {'h':h,'k':k,'l':l}).
        
    Returns:
        Normalized 3D direction vector (np.ndarray, dtype=np.float32).
    """
    pass
```

### `update_dict_recursively`
```python
def update_dict_recursively(base_dict: dict, update_with: dict) -> dict:
    """
    Recursively update a dictionary with another dictionary. Modifies `base_dict` in-place.
    """
    pass
```

### `ensure_directory`
```python
from pathlib import Path
def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary (including parents).
    Returns the Path object.
    """
    pass
```

### `validate_array_shape`
```python
def validate_array_shape(arr: np.ndarray, expected_shape: tuple, name: str) -> None:
    """
    Validate that a NumPy array has the expected shape. Raises ValueError if not.
    """
    pass
```

### `safe_divide`
```python
def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Safely divide two NumPy arrays, element-wise. Handles division by zero by using `fill_value`.
    """
    pass
```

For more detailed documentation of each component, please refer to the individual module documentation files (linked at the top) if they exist, or consult the source code docstrings directly. 