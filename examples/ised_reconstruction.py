#!/usr/bin/env python3
"""
iSED Reconstruction Example

This script demonstrates how to perform inverse SED (iSED) reconstruction
on a molecular dynamics trajectory using the PSA package, including visualization of the
reconstructed motion.
"""

import os
import sys
from pathlib import Path

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
        nz=1   # from hBN_monolayer_config.yaml
    )

    # Perform iSED reconstruction
    print("Performing iSED reconstruction...")
    calculator.ised(
        k_dir_spec='x',  # from hBN_monolayer_config.yaml (ised.k_path.direction)
        k_target=1.7,    # from hBN_monolayer_config.yaml (ised.target_point.k_value)
        w_target=35.5,   # from hBN_monolayer_config.yaml (ised.target_point.w_value_thz)
        char_len_k_path=2.491,  # from hBN_monolayer_config.yaml (md_system.lattice_parameter)
        nk_on_path=250,  # from hBN_monolayer_config.yaml (ised.k_path.n_points)
        bz_cov_ised=4.0, # from hBN_monolayer_config.yaml (ised.k_path.bz_coverage)
        basis_atom_types_ised=[1,2],  # from hBN_monolayer_config.yaml (ised.basis.atom_types)
        rescale_factor='auto',  # from hBN_monolayer_config.yaml (ised.reconstruction.rescaling_factor)
        n_recon_frames=100,  # from hBN_monolayer_config.yaml (ised.reconstruction.num_animation_timesteps)
        dump_filepath=str(output_dir / 'ised_motion.dump'),  # from hBN_monolayer_config.yaml (ised.reconstruction.output_dump_filename)
        plot_dir_ised=output_dir,  # Directory for iSED plots
        plot_max_freq=50.0  # from hBN_monolayer_config.yaml (plotting.max_freq_2d)
    )

    print(f"Analysis complete. Results saved in {output_dir}")
    print("Generated files:")
    print(f"  - {output_dir / 'ised_motion.dump'}")
    # Construct the expected plot filename based on parameters used in calculator.ised()
    k_dir_spec_val = 'x' # Matches the k_dir_spec in the .ised() call
    k_target_val = 1.7   # Matches k_target
    w_target_val = 35.5  # Matches w_target

    k_dir_str = k_dir_spec_val.replace(" ","_").replace("/", "-").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    k_target_str = f"{k_target_val:.2f}".replace('.','p')
    w_target_str = f"{w_target_val:.2f}".replace('.','p')
    expected_plot_filename = f"iSED_{k_dir_str}_{k_target_str}_{w_target_str}.png"
    print(f"  - {output_dir / expected_plot_filename}")

if __name__ == "__main__":
    main() 