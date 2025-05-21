#!/usr/bin/env python3
"""
Basic SED Analysis Example

This script demonstrates how to perform a basic SED analysis
on a molecular dynamics trajectory using the PSA package.
"""

import os
import sys
from pathlib import Path
import numpy as np

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

    # Initialize calculator
    print("Initializing SED calculator...")
    calculator = SEDCalculator(
        traj=trajectory,
        nx=60,  # from hBN_monolayer_config.yaml
        ny=60,  # from hBN_monolayer_config.yaml
        nz=1,   # from hBN_monolayer_config.yaml
        use_velocities=True # from hBN_monolayer_config.yaml (general.use_velocities)
    )

    # Get k-path for Gamma-X direction
    print("Generating k-path...")
    k_mags, k_vecs = calculator.get_k_path(
        direction_spec=[1,0,0], # from hBN_monolayer_config.yaml (sed_calculation.directions)
        bz_coverage=4.0, # from hBN_monolayer_config.yaml (sed_calculation.bz_coverage)
        n_k=250, # from hBN_monolayer_config.yaml (sed_calculation.n_kpoints)
        lat_param=2.491 # from hBN_monolayer_config.yaml (md_system.lattice_parameter)
    )

    # Calculate SED
    print("Calculating SED...")
    sed_data, freqs, is_complex = calculator.calculate(
        k_mags, 
        k_vecs,
        basis_atom_types=[1,2], # from hBN_monolayer_config.yaml (sed_calculation.basis.atom_types)
        summation_mode='coherent' # Or 'incoherent' to test
        )

    # Create SED object
    print("Creating SED object...")
    sed = SED(
        sed=sed_data,
        freqs=freqs,
        k_points=k_mags,
        k_vectors=k_vecs,
        is_complex=is_complex # Pass the flag
    )

    # Generate 2D intensity plot
    print("Generating plot...")
    plotter = SEDPlotter(
        sed,  # sed_obj
        '2d_intensity',  # plot_type
        str(output_dir / 'sed_intensity_2D_100.png'),  # output_path
        title='SED Intensity [100]',
        direction_label='[100]',
        max_freq=50.0,
        log_intensity=True, # Or True, depending on preference
        vmin_percentile=0.0,  # Use actual min for vmin
        vmax_percentile=100.0, # Use actual max for vmax
        theme='light' # Specify light theme (default), or use 'dark' for dark mode
    )
    plotter.generate_plot()

    print(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main() 
