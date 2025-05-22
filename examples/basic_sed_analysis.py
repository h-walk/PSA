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
    output_dir = Path("Si_output")
    output_dir.mkdir(exist_ok=True)

    # Load trajectory
    print("Loading trajectory...")
    loader = TrajectoryLoader(
        filename="Si.lammpstrj",  # Updated filename
        dt=0.02,  # timestep in ps - from hBN_monolayer_config.yaml
        file_format="lammps" # from hBN_monolayer_config.yaml
    )
    trajectory = loader.load()

    # Initialize calculator
    print("Initializing SED calculator...")
    calculator = SEDCalculator(
        traj=trajectory,
        nx=50,  # from hBN_monolayer_config.yaml
        ny=50,  # from hBN_monolayer_config.yaml
        nz=5   # from hBN_monolayer_config.yaml
    )

    # Get k-path for Gamma-X direction
    print("Generating k-path...")
    k_mags, k_vecs = calculator.get_k_path(
        direction_spec=[1,1,0], # from hBN_monolayer_config.yaml (sed_calculation.directions)
        bz_coverage=4.0, # from hBN_monolayer_config.yaml (sed_calculation.bz_coverage)
        n_k=250, # from hBN_monolayer_config.yaml (sed_calculation.n_kpoints)
        lat_param=None # from hBN_monolayer_config.yaml (md_system.lattice_parameter)
    )

    # Calculate SED
    print("Calculating SED...")
    sed = calculator.calculate(
        k_points_mags=k_mags, 
        k_vectors_3d=k_vecs,
        basis_atom_types=None, # from hBN_monolayer_config.yaml (sed_calculation.basis.atom_types)
        summation_mode='coherent', # Or 'incoherent' to test
        k_grid_shape=None # This is a k-path, so no grid shape
    )

    # Generate 2D intensity plot
    print("Generating plot...")
    plotter = SEDPlotter(
        sed,  # sed_obj
        '2d_intensity',  # plot_type
        str(output_dir / 'sed_intensity_2D_110.png'),  # output_path
        title='SED Intensity [110]',
        direction_label='[110]',
        max_freq=None,
        intensity_scale='dsqrt',
        vmin_percentile=0.0,  # Use actual min for vmin
        vmax_percentile=100.0, # Use actual max for vmax
        theme='light' # Specify light theme (default), or use 'dark' for dark mode
    )
    plotter.generate_plot()

    print(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main() 
