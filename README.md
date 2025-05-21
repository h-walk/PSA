# Spectral Displacement Analysis (SDA)

A Python package for analyzing molecular dynamics trajectories using Spectral Displacement Analysis (SDA) and Inverse SED (iSED) reconstruction. This is the `SDA_modular` version.

## Overview

SDA is a powerful tool for analyzing vibrational properties of materials from molecular dynamics simulations. This package provides:

- Spectral Displacement Analysis (SDA) for calculating phonon dispersion relations.
- Chiral SED analysis for studying chiral phonon modes.
- Inverse SED (iSED) reconstruction for visualizing phonon modes.
- Comprehensive visualization tools for 2D and 3D plots.
- Support for various trajectory file formats (via OVITO integration).
- A command-line interface (CLI) for easy execution with configuration files.
- A modular structure for programmatic use and extension.

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- Matplotlib
- OVITO (Python bindings, for trajectory loading)
- tqdm
- PyYAML (for CLI configuration file usage)

### Installation Steps

1.  **Navigate to the project root directory.**
    This is the directory containing this `README.md` file and the `src/` directory.

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv sda_env
    source sda_env/bin/activate  # On Linux/macOS
    # sda_env\Scripts\activate    # On Windows
    ```

3.  **Install the package in editable mode:**
    This command installs the `sda` package from the `src` directory, allowing you to make changes to the source code that are immediately reflected.
    ```bash
    pip install -e .
    ```

4.  **Install `pytest` for running tests (optional but recommended for development):**
    ```bash
    pip install pytest
    ```

## Quick Start: Programmatic Usage

Here's how you can use the `sda` package programmatically. This example demonstrates a basic SED calculation and plot.

```python
from pathlib import Path
from sda import TrajectoryLoader, SEDCalculator, SED, SEDPlotter

# Ensure you have a trajectory file (e.g., "trajectory.lammpstrj")
# in your working directory or provide the correct path.
# For this example, we'll assume it exists.
# The parameters below should be adjusted for your specific system and simulation.

# Define parameters (these would typically come from your knowledge of the system
# or a configuration file for more complex workflows)
trajectory_file = "monolayer300k.lammpstrj" # Replace with your trajectory. This file is expected to be in the project root or a path specified.
dt_ps = 0.0005           # Timestep in picoseconds (must match your simulation)
nx, ny, nz = 60, 60, 1   # Primitive cells in supercell (e.g., for hBN monolayer)
lat_param = 2.491        # Characteristic lattice parameter (e.g., Angstroms)
use_velocities_cfg = True # Whether to use velocities (True for hBN example)
k_direction = [1,0,0]    # K-path direction (e.g., Gamma-X)
bz_coverage_cfg = 4.0    # Brillouin zone coverage for k-path
n_k_points_cfg = 250     # Number of k-points
atom_types_basis = [1,2] # Atom types for basis (e.g., for hBN)
plot_max_freq = 50.0     # Max frequency for 2D plot (THz)

# Create an output directory
output_dir = Path("sda_readme_output")
output_dir.mkdir(exist_ok=True)

# Load trajectory
print(f"Loading trajectory: {trajectory_file}...")
try:
    loader = TrajectoryLoader(filename=trajectory_file, dt=dt_ps, file_format="lammps")
    trajectory = loader.load()
except FileNotFoundError:
    print(f"ERROR: Trajectory file '{trajectory_file}' not found. Please provide a valid trajectory file.")
    exit()
except Exception as e:
    print(f"Error loading trajectory: {e}")
    exit()

# Initialize SEDCalculator
# nx, ny, nz define the primitive cell repetition in your supercell.
# dt_ps is the MD simulation timestep.
print("Initializing SED calculator...")
calculator = SEDCalculator(
    traj=trajectory,
    nx=nx, ny=ny, nz=nz,
    dt_ps=dt_ps,
    use_velocities=use_velocities_cfg
)

# Define a k-path
# lat_param is crucial for correct k-vector scaling if not using reciprocal lattice vectors directly for k_max.
print("Generating k-path...")
k_mags, k_vecs = calculator.get_k_path(
    direction_spec=k_direction,
    bz_coverage=bz_coverage_cfg,
    n_k=n_k_points_cfg,
    lat_param=lat_param
)

# Calculate SED
# A basis (e.g., specific atom types) can be specified.
print("Calculating SED...")
sed_complex, freqs = calculator.calculate(
    k_points_mags=k_mags,
    k_vectors_3d=k_vecs,
    basis_atom_types=atom_types_basis
)

# Create SED object
print("Creating SED object...")
sed_obj = SED(
    sed=sed_complex,
    freqs=freqs,
    k_points=k_mags,
    k_vectors=k_vecs
)

# Plot results (2D intensity plot)
plot_path = output_dir / "sed_intensity_readme.png"
print(f"Generating plot: {plot_path}...")
plotter = SEDPlotter(
    sed_obj_or_list=sed_obj,
    plot_type='2d_intensity',
    out_path_str=str(plot_path),
    title=f'SDA Intensity Plot ({k_direction} direction)',
    direction_label=str(k_direction),
    max_freq=plot_max_freq
)
plotter.generate_plot()
print(f"Programmatic example finished. Output in {output_dir}")
```

For more detailed programmatic examples, see the scripts in the `examples/` directory.

## Command-Line Interface (CLI)

The `sda` package provides a command-line interface for running analyses using a configuration file.

**Basic Usage:**

After installation (`pip install -e .`), you can run the CLI tool from the project root:

```bash
sda --trajectory path/to/your/trajectory.file --config path/to/your/config.yaml --output-dir path/to/output
```

-   `--trajectory`: Path to your MD trajectory file (e.g., LAMMPS dump, VASP OUTCAR). **(Required)**
-   `--config`: Path to your YAML configuration file. (Optional, uses defaults if not provided)
-   `--output-dir`: Directory where results will be saved (defaults to `sda_output`). (Optional)
-   `--chiral`: Enable chiral SED analysis. This overrides the `chiral_mode_enabled` setting in the configuration file. (Optional flag)
-   `--dt`: Override the MD simulation timestep (in picoseconds) specified in the configuration file. (Optional)
-   `--nk`: Override the number of k-points (`n_kpoints`) for SED calculation specified in the configuration file. (Optional)
-   `--recalculate-sed`: Force recalculation of SED data, even if cached `.npy` files exist. (Optional flag)

Run `sda --help` for a full list of CLI options and their descriptions.

## Configuration File

The CLI uses a YAML file to control various aspects of the calculation and plotting.

**Example `config.yaml` Structure:**

```yaml
general:
  trajectory_file_format: 'auto' # 'lammps', 'vasp_outcar', or 'auto'
  use_velocities: False          # Whether to use velocities (True for chiral or velocity-based SED)
  save_npy_trajectory: True      # Cache trajectory data as .npy files for faster subsequent loads
  save_npy_sed_data: True        # Cache SED results as .npy files
  chiral_mode_enabled: False     # Enable chiral SED analysis (calculates and plots phase)

md_system:
  dt: 0.001                      # Timestep of the MD simulation in picoseconds
  nx: 1                          # Number of primitive cells in x direction of the simulation supercell
  ny: 1                          # Number of primitive cells in y direction
  nz: 1                          # Number of primitive cells in z direction
  lattice_parameter: null        # Characteristic lattice parameter (e.g., in Angstroms) for k-path scaling.
                                 # If null, the norm of the first primitive lattice vector |a1| is used.

sed_calculation:
  directions:                    # List of k-path directions to calculate SED for
    - [1,0,0]                    # Example: Gamma-X direction as a vector
    - 'y'                        # Example: Gamma-Y direction (shorthand for [0,1,0])
    # - {'angle': 45}            # Example: Direction at 45 degrees in XY plane from +x axis
    # - {h: 1, k: 1, l: 0}       # Example: Miller indices [h,k,l] (relative to reciprocal primitive vectors)
  n_kpoints: 100                 # Number of k-points along each specified direction
  bz_coverage: 1.0               # Defines the maximum k-magnitude for the k-path, relative to pi/lattice_parameter.
                                 # E.g., 1.0 covers k from 0 to pi/lat_param.
  polarization_indices_chiral: [0,1] # For chiral SED: indices of polarizations (e.g., x, y) used for phase calculation.
  basis:                         # Basis selection for the main SED calculation
    atom_indices: null           # Optional: list of specific 0-based atom indices [0, 1, 2, ...]. Overrides atom_types if specified.
    atom_types: null             # Optional: list of atom types [1, 2, ...] (used if atom_indices is null).

plotting:
  max_freq_2d: null              # Optional: Clip the maximum frequency in 2D plots (THz).
  highlight_2d_intensity:        # Optional: Highlight a rectangular region in 2D intensity plots
    k_min: null
    k_max: null
    w_min: null
    w_max: null
  enable_3d_dispersion_plot: False # Set to True to generate 3D dispersion plots if multiple directions are computed.
  3d_plot_settings:
    intensity_log_scale: True    # Use log10 scale for intensity in 3D plots.
    intensity_thresh_rel: 0.01   # Relative intensity threshold for culling points in 3D intensity plots.

ised: # Inverse SED Reconstruction Parameters
  apply: False                   # Set to True to perform iSED reconstruction.
  k_path:                        # Defines the k-path for the internal SED calculation within iSED.
    direction: 'x'
    characteristic_length: null  # Angstroms. If null, uses md_system.lattice_parameter.
    n_points: 50
    bz_coverage: null            # If null, uses sed_calculation.bz_coverage.
  target_point:                  # Defines the (k, omega) point for reconstruction.
    k_value: 1.57                # Target k-point value (e.g., in 2pi/Angstrom units)
    w_value_thz: 10.0            # Target frequency value (THz)
  basis:                         # Basis for iSED (atoms used for SED calculation and reconstruction).
    atom_indices: null           # List of indices, or list of lists for groups. Overrides atom_types.
    atom_types: null             # List of types, or list of lists for groups.
  reconstruction:
    rescaling_factor: 'auto'     # 'auto' or a numerical factor (e.g., 10.0) to scale reconstructed motion.
    num_animation_timesteps: 100 # Number of frames in the output LAMMPS dump file.
    output_dump_filename: 'ised_motion.dump' # Name of the output LAMMPS dump file.
```

A more comprehensive example configuration file, `hBN_monolayer_config.yaml`, can be found in the project root directory. This file is used by the example scripts.

## Code Examples

Runnable Python script examples demonstrating various features of the `sda` package can be found in the `examples/` directory:

-   `basic_sed_analysis.py`: Demonstrates standard SED calculation and plotting.
-   `chiral_sed_analysis.py`: Shows how to perform chiral SED.
-   `ised_reconstruction.py`: Illustrates iSED mode reconstruction.
-   `visualization_example.py`: Shows various plotting capabilities.

These examples are pre-configured to use parameters similar to those found in the `hBN_monolayer_config.yaml` file, which should be placed in the main project directory for them to run correctly with the intended data. You may need to adjust file paths within the examples if your data or the config file are located elsewhere.

## Running Tests

To run the test suite, ensure you have `pytest` installed (`pip install pytest`). Then, from the project root directory, simply run:

```bash
pytest
```

## Documentation Structure

The full documentation is a work in progress and can be found in the `docs/` directory:

-   `docs/api/`: Detailed API reference for modules and classes.
-   `docs/examples/`: (Planned) Example scripts and notebooks. (Note: current runnable examples are in `SDA_modular/examples/`)
-   `docs/guides/`: (Planned) User guides for common tasks.
-   `docs/tutorials/`: (Planned) Step-by-step tutorials.

## Features

-   **Trajectory Analysis:**
    -   Support for LAMMPS dump and VASP OUTCAR formats (via OVITO Python library).
-   **SED Calculation**
    -   Regular SED analysis (intensity)
    -   Chiral SED analysis (phase difference)
    -   Customizable k-point sampling and path definitions
    -   Support for different polarization directions (implicitly x,y,z)
-   **Visualization**
    -   2D intensity and phase plots
    -   3D intensity and phase dispersion plots (when multiple k-directions are analyzed)
    -   Customizable plot styles and output
    -   Highlighting specific (k,ω) points or regions in 2D plots
-   **iSED Reconstruction**
    -   Visualization of atomic motion for specific (k,ω) modes
    -   Output of reconstructed motion as LAMMPS dump files
    -   Group-wise iSED for complex structures
    -   Automatic or manual rescaling of reconstructed amplitudes

## Contributing

Contributions are welcome! Please outline your proposed changes in an issue or pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if one is added). 