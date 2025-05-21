# PSA User Guide

This guide provides detailed instructions for using the Phonon Spectral Analysis (PSA) package (`PSA` version).

## Table of Contents

1. [Installation](#installation)
2. [Programmatic Usage (Library)](#programmatic-usage-library)
3. [Command-Line Interface (CLI) Usage](#command-line-interface-cli-usage)
4. [Trajectory Loading](#trajectory-loading)
5. [SED Calculation Details](#sed-calculation-details)
6. [Plotting and Visualization](#plotting-and-visualization)
7. [iSED Reconstruction Details](#ised-reconstruction-details)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

Before installing the `psa` package, ensure you have:

- Python 3.8 or higher
- NumPy
- Matplotlib
- OVITO (Python bindings, required for trajectory loading by `TrajectoryLoader`)
- tqdm (for progress bars)
- PyYAML (if using the CLI with YAML configuration files)

### Installation Steps

1. Navigate to the `PSA` root directory (where `setup.py` is located).
2. Install the package in editable mode. This allows you to use the package and also modify the source code if needed.
    ```bash
    pip install -e .
    ```
3. To run tests (optional):
    ```bash
    pip install pytest
    cd tests
    pytest
    cd ..
    ```

## Programmatic Usage (Library)

This section covers how to use the `psa` package as a Python library in your own scripts.

### Loading a Trajectory

```python
from pathlib import Path
from psa import TrajectoryLoader

# Ensure you have a trajectory file (e.g., "trajectory.lammpstrj")
# For example:
# Path("trajectory.lammpstrj").write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS id type x y z\n1 1 0.5 0.5 0.5\n")

output_dir = Path("user_guide_output")
output_dir.mkdir(exist_ok=True)

# Initialize loader
# Replace "trajectory.lammpstrj" with your actual trajectory file path.
# dt should be the timestep of your MD simulation in picoseconds.
loader = TrajectoryLoader(
    filename="trajectory.lammpstrj",
    dt=0.001,  # Timestep in ps
    file_format="lammps"  # or 'vasp_outcar', 'auto' for automatic detection
)

# Load trajectory
trajectory = loader.load()
print(f"Trajectory loaded: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms.")
```

### Basic SED Analysis

```python
from psa import SEDCalculator, SED # Assuming 'trajectory' is loaded from previous step

# Initialize calculator
# nx, ny, nz: number of primitive cells the simulation box corresponds to.
calculator = SEDCalculator(
    traj=trajectory,
    nx=1,  # Adjust based on your system supercell
    ny=1,  # Adjust based on your system supercell
    nz=1,  # Adjust based on your system supercell
    dt_ps=trajectory.timesteps[1] - trajectory.timesteps[0] if trajectory.n_frames > 1 else 0.001 # dt_ps from loader or traj
)

# Get k-path (e.g., Gamma-X direction)
# lat_param can be specified, otherwise it's inferred from calculator.a1
k_mags, k_vecs = calculator.get_k_path(
    direction_spec=[1,0,0],  # Example: [1,0,0] direction string or vector
    bz_coverage=1.0,        # Coverage of Brillouin zone (e.g., 1.0 for one BZ length)
    n_k=50,                 # Number of k-points
    # lat_param=2.5 # Optional: characteristic lattice parameter for k-path units
)

# Calculate SED
sed_complex, freqs = calculator.calculate(
    k_points_mags=k_mags, # Magnitudes (for reference, plotting)
    k_vectors_3d=k_vecs   # Actual 3D k-vectors used in calculation
)

# Create SED object for easier handling and plotting
sed_data_obj = SED(
    sed=sed_complex,
    freqs=freqs,
    k_points=k_mags,
    k_vectors=k_vecs
)
print("SED calculation complete.")
```

### Visualization

```python
from psa import SEDPlotter # Assuming 'sed_data_obj' and 'output_dir' exist

# Create 2D intensity plot
plot_path = output_dir / "sed_intensity_user_guide.png"
plotter = SEDPlotter(
    sed_obj_or_list=sed_data_obj,
    plot_type='2d_intensity',
    out_path_str=str(plot_path),
    title='SED Intensity ([100] direction)',
    direction_label='[100]'
)
plotter.generate_plot()
print(f"Plot saved to {plot_path}")
```

## Command-Line Interface (CLI) Usage

The `psa` package includes a command-line interface (CLI) for running analyses using a configuration file. After installing the package, you can invoke the CLI as follows:

```bash
psa --trajectory path/to/your/trajectory.file --config path/to/your/config.yaml --output-dir path/to/output_directory
```

- `--trajectory`: Path to your MD trajectory file.
- `--config`: Path to your YAML configuration file.
- `--output-dir` (optional): Directory to save results (defaults to `psa_output`).

Refer to the main `README.md` or `docs/api/README.md` for a detailed example of the `config.yaml` structure. The CLI script (`cli.py`) parses this file and runs the appropriate analysis steps.

## Trajectory Loading

(`psa.io.TrajectoryLoader`)

### Supported File Formats

The `TrajectoryLoader` uses OVITO in the backend for parsing trajectory files. Therefore, it supports formats recognized by OVITO, primarily:

- LAMMPS trajectory files (typically `.lammpstrj` or custom dump files).
- VASP OUTCAR files.
- Other formats if OVITO can parse them (e.g., XYZ, PDB, etc., though velocities might be missing).

Specify `file_format` as 'lammps', 'vasp_outcar', or 'auto' during `TrajectoryLoader` initialization.

### Caching
To speed up subsequent loading of the same trajectory, `TrajectoryLoader` can save the parsed data (positions, velocities, types, box matrix) into `.npy` files in the same directory as the trajectory file. This is controlled by the `save_npy_trajectory` option in the CLI config or can be implemented manually after loading.

## SED Calculation Details

(`psa.core.SEDCalculator`)

### Basis Atom Selection
For SED calculations, you can choose to include all atoms (default) or a subset:

1. **By Atom Indices**: Pass a list or NumPy array of 0-based atom indices to the `basis_atom_indices` parameter of the `calculate` method.
    ```python
    # Example: Calculate SED using only the first 10 atoms
    # sed_complex, freqs = calculator.calculate(k_mags, k_vecs, basis_atom_indices=list(range(10)))
    ```

2. **By Atom Types**: If `basis_atom_indices` is not provided, the CLI (and potentially your custom script logic) might use `basis_atom_types` from a config to select atoms. The `SEDCalculator.calculate` method itself only takes indices. You would typically get these indices by filtering `trajectory.types`.
    ```python
    # Example: Get indices for atoms of type 1 or 2
    # type_mask = np.isin(trajectory.types, [1, 2])
    # type_indices = np.where(type_mask)[0]
    # sed_complex, freqs = calculator.calculate(k_mags, k_vecs, basis_atom_indices=type_indices)
    ```

### Regular SED (Intensity)
The standard SED calculation yields complex spectral data. The intensity is typically `np.sum(np.abs(sed_complex)**2, axis=-1)`.

### Chiral SED (Phase)
For chiral SED analysis:
1. Ensure `use_velocities=True` when initializing `SEDCalculator` if your chiral definition requires it (often based on velocity cross-products, although the current `calculate_chiral_phase` works on any two complex signals).
2. Calculate the regular complex SED data (`sed_complex`).
3. Use the `calculate_chiral_phase` method, providing two polarization components of the `sed_complex` data:
    ```python
    # sed_complex has shape (n_freqs, n_k_points, 3_polarizations)
    # Example: phase between 0th (e.g., x) and 1st (e.g., y) polarizations
    # phase_data = calculator.calculate_chiral_phase(
    #     Z1=sed_data_obj.sed[:,:,0],
    #     Z2=sed_data_obj.sed[:,:,1],
    #     angle_range_opt="C"  # Option for phase range, "C" is default
    # )
    # sed_data_obj.phase = phase_data # Store it in the SED object
    ```

## Plotting and Visualization

(`psa.visualization.SEDPlotter`)

### 2D Plots

- **Intensity Plot** (`plot_type='2d_intensity'`):
    Visualizes SED intensity \( I(k, \omega) \) as a colormap.
    ```python
    # plotter = SEDPlotter(
    #     sed_obj_or_list=sed_data_obj,
    #     plot_type='2d_intensity',
    #     out_path_str=str(output_dir / 'intensity_2d.png'),
    #     title='My SED Intensity',
    #     cmap='inferno' # Optional: specify colormap
    # )
    # plotter.generate_plot()
    ```

- **Phase Plot** (`plot_type='2d_phase'`):
    Visualizes chiral phase difference \( \Delta\phi(k, \omega) \) if available in the `SED` object.
    ```python
    # plotter = SEDPlotter(
    #     sed_obj_or_list=sed_data_obj_with_phase, # Assumes sed_data_obj_with_phase.phase is populated
    #     plot_type='2d_phase',
    #     out_path_str=str(output_dir / 'phase_2d.png')
    # )
    # plotter.generate_plot()
    ```

### 3D Plots
For visualizing dispersion over a 2D k-space area (e.g., a plane in the BZ), you would typically run SED calculations for multiple k-paths (directions) that scan this area. Collect the resulting `SED` objects into a list.

- **3D Intensity Plot** (`plot_type='3d_intensity'`):
    ```python
    # sed_results_list = [sed_obj_dir1, sed_obj_dir2, ...] # List of SED objects from different k-paths
    # plotter_3d = SEDPlotter(
    #     sed_obj_or_list=sed_results_list,
    #     plot_type='3d_intensity',
    #     out_path_str=str(output_dir / 'intensity_3d.png')
    # )
    # plotter_3d.generate_plot()
    ```
- **3D Phase Plot** (`plot_type='3d_phase'`): Similar, but for phase data.

### Plot Customization
`SEDPlotter` accepts various `**kwargs` for customization, such as `title`, `direction_label`, `cmap`, `max_freq`, `highlight_region`. Refer to `SEDPlotter` docstrings or source for details.

## iSED Reconstruction Details

(`psa.core.SEDCalculator.ised` method)

iSED reconstructs and visualizes atomic motion corresponding to a specific \( (k, \omega) \) mode.

```python
# Assuming 'calculator' and 'output_dir' are initialized
# calculator.ised(
#     k_dir_spec='x',  # Direction of the k-path for internal SED calculation
#     k_target=1.57,    # Target k-point magnitude (2π/Å) for reconstruction
#     w_target=10.0,   # Target frequency (THz) for reconstruction
#     char_len_k_path=trajectory.box_lengths[0] / calculator.nx,  # Characteristic length (e.g., primitive cell length in Å)
#     nk_on_path=100,  # Number of k-points for internal SED calculation
#     bz_cov_ised=1.0, # BZ coverage for internal SED
#     basis_atom_types_ised=None, # Example: [1, 2] to use atoms of type 1 and 2. None for all atoms.
#                                 # Can be list of lists for multiple groups: [[1],[2]]
#     rescale_factor='auto',  # 'auto' or a float value for amplitude rescaling
#     n_recon_frames=100,  # Number of frames in output animation dump
#     dump_filepath=str(output_dir / 'ised_reconstructed_motion.dump'),
#     plot_dir_ised=output_dir, # Directory to save plot of iSED input spectrum
#     plot_max_freq=20.0   # Max frequency for the iSED input spectrum plot (THz)
# )
```
Key parameters for `ised`:
- `k_dir_spec`, `k_target`, `w_target`: Define the mode to reconstruct.
- `char_len_k_path`, `nk_on_path`, `bz_cov_ised`: Control the internal SED calculation used to find the amplitude and phase of the target mode.
- `basis_atom_idx_ised` or `basis_atom_types_ised`: Select atoms for the reconstruction. Can be a flat list for a single group or a list of lists for multiple independent groups.
- `rescale_factor`: Controls the amplitude of the reconstructed wiggles.
- `dump_filepath`: Output file for the animated mode (LAMMPS dump format).
- `plot_dir_ised`, `plot_max_freq`: For plotting the summed SED intensity along the `k_dir_spec`, highlighting the target mode.

## Troubleshooting

- **`FileNotFoundError` for trajectory**: Ensure the path to your trajectory file is correct.
- **OVITO import errors**: Make sure OVITO Python bindings are correctly installed and accessible in your Python environment.
- **Memory issues**: Large trajectories or a high number of k-points/frequencies can consume significant memory. Consider downsampling your trajectory or analyzing smaller k-space regions if needed.
- **Zero SED intensity**:
    - Check `dt_ps` matches your simulation timestep.
    - Ensure `nx, ny, nz` correctly define the primitive cell repeat units in your supercell.
    - Verify atom basis selection if not using all atoms.
    - Small `bz_coverage` or very few `n_k` points might miss features.
- **iSED issues**:
    - Ensure `char_len_k_path` is appropriate (usually a primitive lattice parameter).
    - Target (k,w) might not correspond to a strong mode. Check the iSED input plot. 