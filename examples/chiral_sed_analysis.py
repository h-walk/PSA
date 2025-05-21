#!/usr/bin/env python3
"""
Chiral SED Analysis Example

This script demonstrates how to perform chiral SED analysis
on a molecular dynamics trajectory using the PSA package, including phase calculation and visualization.
"""

import os
import sys
from pathlib import Path
import numpy as np # Added for frequency printout

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from psa import TrajectoryLoader, SEDCalculator, SED, SEDPlotter

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Load trajectory
    print("Loading trajectory...")
    loader = TrajectoryLoader(
        filename="monolayer300k.lammpstrj",  # Updated filename
        dt=0.005,  # timestep in ps - from hBN_monolayer_config.yaml
        file_format="lammps" # from hBN_monolayer_config.yaml
    )
    trajectory = loader.load()

    # Initialize calculator with velocities enabled for chiral analysis
    print("Initializing SED calculator with chiral mode...")
    calculator = SEDCalculator(
        traj=trajectory,
        nx=60,  # from hBN_monolayer_config.yaml
        ny=60,  # from hBN_monolayer_config.yaml
        nz=1,   # from hBN_monolayer_config.yaml
        use_velocities=True  # from hBN_monolayer_config.yaml (general.use_velocities)
    )

    # Get k-path for Gamma-X direction
    print("Generating k-path...")
    k_mags, k_vecs = calculator.get_k_path(
        direction_spec=[1,0,0], # from hBN_monolayer_config.yaml (sed_calculation.directions)
        bz_coverage=4.0, # from hBN_monolayer_config.yaml (sed_calculation.bz_coverage)
        n_k=250, # from hBN_monolayer_config.yaml (sed_calculation.n_kpoints)
        lat_param=2.491 # from hBN_monolayer_config.yaml (md_system.lattice_parameter)
    )

    # Calculate SED with chiral phase
    print("Calculating SED...")
    # For chiral phase calculation, we need the complex SED components directly.
    # So, coherent summation is implied for the components used in chiral phase.
    sed_complex_data, freqs, _ = calculator.calculate(
        k_mags, 
        k_vecs,
        basis_atom_types=[1,2], # This will be a coherent sum over types 1 and 2
        summation_mode='coherent' # Must be coherent for subsequent chiral phase calculation
        )

    # Calculate phase angle
    print("Calculating phase angle...")
    phase = calculator.calculate_chiral_phase(
        Z1=sed_complex_data[:,:,0],
        Z2=sed_complex_data[:,:,1],
        angle_range_opt="C"
    )

    # Create SED object with phase
    # The sed_complex_data is stored for potential use, but phase plot uses sed.phase
    # Intensity plot will be based on this sed_complex_data.
    print("Creating SED object with phase data...")
    sed = SED(
        sed=sed_complex_data,
        freqs=freqs,
        k_points=k_mags,
        k_vectors=k_vecs,
        phase=phase,
        is_complex=True # Data fed to chiral phase and stored is complex
    )

    # Generate intensity plot
    print("Generating intensity plot...")
    intensity_plotter = SEDPlotter(
        sed, # sed_obj (positional)
        '2d_intensity', # plot_type (positional)
        str(output_dir / 'chiral_sed_intensity_2D.png'), # output_path (positional)
        title='SED Intensity [100] (Chiral Example)',
        direction_label='[100]',
        max_freq=50.0,
        log_intensity=False, # Explicitly set, can be True for log scale
        vmin_percentile=1.0,
        vmax_percentile=99.0
    )
    intensity_plotter.generate_plot()

    # Generate phase plot
    # For phase plots, vmin/vmax often fixed e.g. -pi/2 to pi/2 or 0 to pi
    # The _plot_2d_phase method has defaults: vmin=-np.pi/2, vmax=np.pi/2
    # Default cmap 'inferno' might not be ideal for phase, 'coolwarm' or 'hsv' often used.
    print("Generating phase plot...")
    phase_plotter = SEDPlotter(
        sed, # sed_obj (positional)
        '2d_phase', # plot_type (positional)
        str(output_dir / 'chiral_sed_phase_2D.png'), # output_path (positional)
        title='Chiral Phase [100]',
        direction_label='[100]',
        max_freq=50.0,
        cmap='coolwarm' # Example: using a different cmap for phase
    )
    phase_plotter.generate_plot()

    print(f"Analysis complete. Results saved in {output_dir}")
    print("Generated files:")
    print(f"  - {output_dir / 'chiral_sed_intensity_2D.png'}")
    print(f"  - {output_dir / 'chiral_sed_phase_2D.png'}")

if __name__ == "__main__":
    main() 
