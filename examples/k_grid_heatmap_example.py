#!/usr/bin/env python3
"""
K-Grid Heatmap Example for SED Analysis

This script demonstrates how to generate and plot SED data on a 2D k-grid,
visualizing it as heatmaps for specific frequencies using the PSA package.
It also shows how to save and load the calculated SED data to avoid recomputation,
and how to use a global intensity scale for relative comparisons.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from psa import TrajectoryLoader, SEDCalculator, SED, SEDPlotter

def main():
    # --- Configuration ---
    base_data_path = Path(".") 
    trajectory_file = "monolayer300k.lammpstrj"
    dt_ps = 0.005
    file_format = "lammps"
    nx, ny, nz = 60, 60, 1
    use_velocities = True
    basis_atom_types = [1, 2]
    summation_mode = 'coherent'

    # K-grid parameters
    plane_to_plot = "xy"
    k_range_comp1 = (-6, 6)  # Range for kx if plane is 'xy' (in 2pi/Angstrom)
    k_range_comp2 = (-6, 6)  # Range for ky if plane is 'xy' (in 2pi/Angstrom)
    n_k_comp1 = 241          # Number of points for kx
    n_k_comp2 = 241          # Number of points for ky
    k_fixed_val = 0.0        # Value for kz if plane is 'xy'

    # Frequencies to plot: 0 to 50 THz in 0.25 THz increments
    freq_start_thz = 0.0
    freq_stop_thz = 50.0
    freq_step_thz = 0.25
    frequencies_to_plot_thz = np.arange(freq_start_thz, freq_stop_thz + freq_step_thz/2, freq_step_thz)
    apply_log_intensity_scale = True # Set to False for linear intensity scale

    output_dir_name = "output_k_grid_heatmaps"
    saved_sed_basename = "sed_k_grid_data" # Basename for saved SED .npy files
    # ---

    # Create output directory
    output_dir = Path(output_dir_name)
    output_dir.mkdir(exist_ok=True)

    sed_data_path_base = Path("output_k_grid_heatmaps") / saved_sed_basename # Load from previous example's output if available
    sed_freqs_file = sed_data_path_base.with_suffix('.freqs.npy')

    if sed_freqs_file.exists():
        print(f"Found existing SED data at {sed_data_path_base}. Loading...")
        try:
            sed_obj = SED.load(sed_data_path_base)
            print("SED data loaded successfully.")
            if sed_obj.k_grid_shape != (n_k_comp1, n_k_comp2):
                print(f"Warning: Loaded k_grid_shape {sed_obj.k_grid_shape} differs from script parameters ({n_k_comp1}, {n_k_comp2}).")
                print("This might lead to unexpected behavior if script parameters for grid are not aligned with loaded data.")
        except Exception as e:
            print(f"Error loading SED data: {e}. Will attempt to recalculate.")
            sed_obj = None
    else:
        print(f"No existing SED data found at {sed_data_path_base}. Will calculate.")
        sed_obj = None

    if sed_obj is None:
        print("Recalculating SED data...")
        # (Re)Create the output directory for saving if recalculating, in case it's different
        calc_output_dir = Path("output_k_grid_heatmaps") # Original save location
        calc_output_dir.mkdir(exist_ok=True)
        sed_data_path_base_for_saving = calc_output_dir / saved_sed_basename

        print(f"Loading trajectory from {base_data_path / trajectory_file}...")
        loader = TrajectoryLoader(
            filename=str(base_data_path / trajectory_file),
            dt=dt_ps,
            file_format=file_format
        )
        try:
            trajectory = loader.load()
        except FileNotFoundError:
            print(f"ERROR: Trajectory file not found at {base_data_path / trajectory_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            sys.exit(1)

        print("Initializing SED calculator...")
        calculator = SEDCalculator(
            traj=trajectory,
            nx=nx,
            ny=ny,
            nz=nz,
            use_velocities=use_velocities
        )

        print(f"Generating k-grid for plane: {plane_to_plot}...")
        k_mags_dummy, k_vecs_grid, k_grid_shape = calculator.get_k_grid(
            plane=plane_to_plot,
            k_range_x=k_range_comp1, 
            k_range_y=k_range_comp2,
            n_kx=n_k_comp1,
            n_ky=n_k_comp2,
            k_fixed_val=k_fixed_val
        )
        print(f"  k_grid_shape: {k_grid_shape}, total k-points: {k_vecs_grid.shape[0]}")

        print("Calculating SED on the k-grid...")
        sed_data_calc, freqs_calc, is_complex_calc = calculator.calculate(
            k_points_mags=k_mags_dummy,
            k_vectors_3d=k_vecs_grid,
            basis_atom_types=basis_atom_types,
            summation_mode=summation_mode
        )

        print("Creating SED object with k-grid information...")
        sed_obj = SED(
            sed=sed_data_calc,
            freqs=freqs_calc,
            k_points=k_mags_dummy,
            k_vectors=k_vecs_grid,
            k_grid_shape=k_grid_shape,
            is_complex=is_complex_calc
        )
        
        print(f"Saving calculated SED data to {sed_data_path_base_for_saving}...")
        sed_obj.save(sed_data_path_base_for_saving)
        print("SED data saved.")

    # Determine global vmin and vmax for consistent color scaling
    global_vmin, global_vmax = None, None
    if sed_obj is not None:
        all_intensities = sed_obj.intensity # Shape: (n_freqs, total_k_points_in_grid)
        if all_intensities.size > 0:
            if apply_log_intensity_scale:
                positive_intensities = all_intensities[all_intensities > 1e-12]
                if positive_intensities.size > 0:
                    global_vmax = np.log10(np.max(positive_intensities))
                    global_vmin = np.log10(np.min(positive_intensities))
                else: # All intensities are too small or zero
                    global_vmin, global_vmax = -12, -11 # Default log range for very small/zero data
                    print("Warning: All intensity data is zero or too small for log scale. Using default log range.")
            else: # Linear scale
                global_vmin = np.min(all_intensities)
                global_vmax = np.max(all_intensities)
            
            if global_vmin is not None and global_vmax is not None and global_vmin == global_vmax:
                print("Warning: Calculated global_vmin and global_vmax are equal (flat data). Adjusting slightly.")
                global_vmin = global_vmin - 0.1 if global_vmin != 0 else -0.1
                global_vmax = global_vmax + 0.1 if global_vmax != 0 else 0.1
        else:
            print("Warning: No intensity data found in SED object. Plotter will use automatic scaling.")

    # Generate 3D heatmap plots
    if sed_obj is not None:
        if global_vmin is not None and global_vmax is not None:
            print(f"Generating '3d_heatmap' plots with determined global scale (vmin={global_vmin:.2f}, vmax={global_vmax:.2f}) for plane {plane_to_plot.upper()}...")
        else:
            print(f"Warning: Global vmin/vmax could not be determined. Plotter will use its automatic scaling for plane {plane_to_plot.upper()}...")

        for target_freq in frequencies_to_plot_thz:
            plot_filename = f"sed_heatmap_{plane_to_plot}_freq_{target_freq:.2f}THz.png" # Simplified filename, removed _scale
            plot_output_path = output_dir / plot_filename.replace('.', 'p')
            
            print(f"  Plotting for {target_freq:.2f} THz, saving to {plot_output_path}...")
            
            plotter = SEDPlotter(
                sed_obj=sed_obj,
                plot_type='3d_heatmap',
                output_path=str(plot_output_path),
                title=rf'SED Heatmap ({plane_to_plot.upper()} plane, {k_fixed_val} $2\pi/\AA$)',
                heatmap_plane=plane_to_plot,
                heatmap_target_freq_thz=target_freq,
                log_intensity=apply_log_intensity_scale, # This controls if SEDPlotter also applies log if vmin/vmax are None
                cmap='inferno',
                vmin=global_vmin, # Pass determined global_vmin (can be None)
                vmax=global_vmax, # Pass determined global_vmax (can be None)
                theme='light',
                grid=False
            )
            plotter.generate_plot()
        print(f"Analysis complete. Heatmap plots saved in {output_dir}")
    else: # sed_obj is None
        print("Error: SED object is not available. Plotting cannot proceed.")


if __name__ == "__main__":
    main() 
