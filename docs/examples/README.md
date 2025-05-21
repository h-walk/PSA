# SDA Examples Snippets

This document provides illustrative code snippets for using the `sda` package. For complete, runnable example scripts, please refer to the files in the `SDA_modular/examples/` directory at the root of the project.

## Basic Operations

These snippets assume you have already loaded a trajectory into a `trajectory` variable (see `docs/guides/user_guide.md` or `SDA_modular/examples/basic_sed_analysis.py`).

### 1. Basic SED Calculation and Plotting

```python
from pathlib import Path # For output paths
import numpy as np # For np.isin if used in atom selection later
from sda import TrajectoryLoader, SEDCalculator, SED, SEDPlotter

# --- Assume trajectory is loaded as 'trajectory' ---
# Example:
# loader = TrajectoryLoader(filename="trajectory.lammpstrj", dt=0.001)
# trajectory = loader.load()
# output_dir = Path("docs_example_output")
# output_dir.mkdir(exist_ok=True)
# ----------------------------------------------------

# Initialize calculator
# Adjust nx, ny, nz, and dt_ps according to your system and simulation
calculator = SEDCalculator(
    traj=trajectory,
    nx=1, ny=1, nz=1,
    dt_ps=0.001 # Ensure this matches your trajectory timestep in ps
)

# Get k-path (e.g., Gamma-X direction)
k_mags, k_vecs = calculator.get_k_path(
    direction_spec=[1,0,0], # Can be string like 'x', list, dict, etc.
    bz_coverage=1.0,        # How far along the BZ direction
    n_k=50                  # Number of k-points
    # lat_param=2.5 # Optional: characteristic lattice parameter for k-path length
)

# Calculate SED
sed_complex, freqs = calculator.calculate(
    k_points_mags=k_mags, # k-magnitudes (for reference/plotting)
    k_vectors_3d=k_vecs   # Actual 3D k-vectors for calculation
)

# Create SED object
sed_data_obj = SED(
    sed=sed_complex,
    freqs=freqs,
    k_points=k_mags,
    k_vectors=k_vecs
)

# Generate 2D intensity plot
plot_file_path = output_dir / "docs_sed_intensity_2D.png"
plotter = SEDPlotter(
    sed_obj_or_list=sed_data_obj,
    plot_type='2d_intensity',
    out_path_str=str(plot_file_path),
    title='SED Intensity Example ([100])',
    direction_label='[100]'
)
plotter.generate_plot()
print(f"Plot generated: {plot_file_path}")
```

### 2. Chiral SED Analysis

This example shows how to perform chiral SED analysis. It builds upon the previous SED calculation.

```python
# --- Assume 'calculator' and 'sed_data_obj' from previous snippet ---
# If chiral analysis requires velocities, ensure SEDCalculator was initialized with use_velocities=True
# calculator_chiral = SEDCalculator(traj=trajectory, nx=1,ny=1,nz=1, dt_ps=0.001, use_velocities=True)
# sed_complex_for_chiral, freqs_for_chiral = calculator_chiral.calculate(k_mags, k_vecs)
# sed_data_for_chiral_obj = SED(sed_complex_for_chiral, freqs_for_chiral, k_mags, k_vecs)
# -------------------------------------------------------------------

# Calculate chiral phase (e.g., between 0th and 1st polarization components)
# Ensure sed_data_obj.sed has at least two polarization components
if sed_data_obj.sed.shape[-1] >= 2:
    phase_data = calculator.calculate_chiral_phase(
        Z1=sed_data_obj.sed[:,:,0], # First polarization component
        Z2=sed_data_obj.sed[:,:,1], # Second polarization component
        angle_range_opt="C"         # Default phase calculation method
    )
    sed_data_obj.phase = phase_data # Store phase in the SED object

    # Generate phase plot
    phase_plot_path = output_dir / "docs_sed_phase_2D.png"
    phase_plotter = SEDPlotter(
        sed_obj_or_list=sed_data_obj,
        plot_type='2d_phase',
        out_path_str=str(phase_plot_path),
        title='Chiral Phase Example ([100])',
        direction_label='[100]'
    )
    phase_plotter.generate_plot()
    print(f"Phase plot generated: {phase_plot_path}")
else:
    print("Skipping chiral phase plot: SED data has < 2 polarizations.")
```

### 3. iSED Reconstruction

This example demonstrates how to perform iSED reconstruction.

```python
# --- Assume 'calculator' and 'output_dir' are defined ---

# Perform iSED reconstruction
# Ensure char_len_k_path is appropriate (e.g., a primitive lattice parameter)
char_length = np.linalg.norm(calculator.a1) # Example: use norm of first primitive vector
ised_dump_path = output_dir / "docs_ised_motion.dump"

calculator.ised(
    k_dir_spec='x',            # Direction for the internal k-path scan
    k_target=1.57,             # Target k-point magnitude (2π/Å) for reconstruction
    w_target=10.0,             # Target frequency (THz) for reconstruction
    char_len_k_path=char_length, # Characteristic length (Å) for k-path units
    nk_on_path=50,             # Number of k-points for internal SED scan
    bz_cov_ised=1.0,           # BZ coverage for internal SED scan
    basis_atom_types_ised=None, # Use all atoms. Or e.g., [1,2] for types 1 and 2
                                 # Or list of lists for groups: [[1], [2]]
    rescale_factor='auto',      # Automatic amplitude rescaling, or a float value
    n_recon_frames=100,        # Number of frames in the output animation
    dump_filepath=str(ised_dump_path), # Output LAMMPS dump file path
    plot_dir_ised=output_dir,  # Directory to save a plot of the iSED input spectrum
    plot_max_freq=20.0         # Max frequency for the iSED input spectrum plot (THz)
)
print(f"iSED reconstruction dump saved to: {ised_dump_path}")
```

## Advanced Usage Snippets

### 1. Multiple Directions for 3D Dispersion

This shows how to analyze multiple directions and create a 3D dispersion plot.

```python
# --- Assume 'calculator' and 'output_dir' are defined ---

directions_to_scan = [
    {"direction_spec": [1,0,0], "label": "Gamma-X"},
    {"direction_spec": [0,1,0], "label": "Gamma-Y"},
    # Add more directions as needed, e.g., for a 2D k-plane scan
    # For a true 2D k-plane scan, k_vectors would need to form a 2D grid.
    # This example just computes along lines.
]

sed_results_list = []
for item in directions_to_scan:
    k_mags_adv, k_vecs_adv = calculator.get_k_path(
        direction_spec=item["direction_spec"],
        bz_coverage=1.0,
        n_k=30 # Fewer points for quicker example
    )
    sed_complex_adv, freqs_adv = calculator.calculate(k_mags_adv, k_vecs_adv)
    # Note: For true 3D plotting of a k-plane, k_vectors in SED object
    # should represent the actual 3D k-points, not just magnitudes along a line.
    # The current SED.gather_3d() uses SED.k_vectors which are 3D.
    sed_obj_adv = SED(sed_complex_adv, freqs_adv, k_mags_adv, k_vecs_adv)
    sed_results_list.append(sed_obj_adv)

# Generate 3D dispersion plot if data was collected
if sed_results_list:
    plot3d_path = output_dir / "docs_disp_3D_sed_intensity.png"
    plotter_3d = SEDPlotter(
        sed_obj_or_list=sed_results_list,
        plot_type='3d_intensity',
        out_path_str=str(plot3d_path),
        title='3D SED Dispersion Example',
        intensity_log_scale=True
    )
    plotter_3d.generate_plot()
    print(f"3D plot generated: {plot3d_path}")
```

### 2. Custom Atom Selection for SED Calculation

Demonstrates selecting specific atoms for an SED calculation.

```python
# --- Assume 'calculator', 'trajectory', 'k_mags', 'k_vecs', 'output_dir' are defined ---

# Example: Select atoms of type 1 (if trajectory.types is populated)
if hasattr(trajectory, 'types') and trajectory.types.size > 0:
    type_mask = np.isin(trajectory.types, [1]) # Select atoms of type 1
    basis_indices_selected = np.where(type_mask)[0]
    if basis_indices_selected.size > 0:
        # Calculate SED for selected atoms
        sed_complex_sel, freqs_sel = calculator.calculate(
            k_points_mags=k_mags,
            k_vectors_3d=k_vecs,
            basis_atom_indices=basis_indices_selected
        )
        # Create and plot SED
        sed_selected_obj = SED(sed_complex_sel, freqs_sel, k_mags, k_vecs)
        plot_selected_path = output_dir / "docs_sed_intensity_selected_atoms.png"
        plotter_selected = SEDPlotter(
            sed_obj_or_list=sed_selected_obj,
            plot_type='2d_intensity',
            out_path_str=str(plot_selected_path),
            title='SED Intensity (Type 1 Atoms)',
            direction_label=str(calculator.get_k_path(direction_spec=[1,0,0], bz_coverage=1.0, n_k=1)[1][0]) # Example label
        )
        plotter_selected.generate_plot()
        print(f"Plot for selected atoms: {plot_selected_path}")
    else:
        print("No atoms of type 1 found to select.")
else:
    print("Trajectory does not have type information for selection example.")
```

## Example CLI Configuration Structure

For command-line usage, you typically provide a YAML configuration file. Here is an example structure (refer to `docs/api/README.md` or the main project `README.md` for a more detailed version with comments):

```yaml
general:
  trajectory_file_format: 'auto'
  use_velocities: False
  # ... and other general settings ...

md_system:
  dt: 0.001
  nx: 1
  ny: 1
  nz: 1
  # ... and other system settings ...

sed_calculation:
  directions:
    - [1,0,0]
  n_kpoints: 100
  # ... and other calculation settings ...

plotting:
  max_freq_2d: null
  # ... and other plotting settings ...

ised:
  apply: False
  # ... and other iSED settings ...
```

This configuration is used by the `sda` command-line tool. See `sda --help` for more details.