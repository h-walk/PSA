"""
Example script demonstrating the visualization capabilities of the PSA package.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from psa import SED, SEDPlotter

def create_sample_sed_data():
    """Create sample SED data for demonstration."""
    # Create sample k-points and frequencies
    k_points = np.linspace(0, 2*np.pi, 100)
    freqs = np.linspace(0, 50, 200)  # 0-50 THz
    
    # Create sample complex SED data
    k_mesh, f_mesh = np.meshgrid(k_points, freqs)
    
    # Create a sample dispersion relation (e.g., acoustic phonon)
    v_sound = 5.0  # Example sound velocity, adjust as needed
    # For a 1D path, k_points are magnitudes. For k_vectors use full 3D vectors if available.
    # Here, k_points are magnitudes along a path.
    # omega = v_sound * k_points # A simple linear dispersion for example
    
    # Let's make a more 2D-like feature for plotting
    # Example: a mode centered at k=pi, freq=15, with some spread
    center_k = np.pi
    center_freq = 15.0
    spread_k = 0.5
    spread_freq = 5.0
    intensity = np.exp(-((k_mesh - center_k)**2 / (2*spread_k**2) + (f_mesh - center_freq)**2 / (2*spread_freq**2)))
    
    # Add another fainter mode
    center_k2 = np.pi / 2
    center_freq2 = 30.0
    intensity += 0.5 * np.exp(-((k_mesh - center_k2)**2 / (2*spread_k**2) + (f_mesh - center_freq2)**2 / (2*spread_freq**2)))
    
    # Add some noise
    intensity += 0.1 * np.random.randn(*intensity.shape)
    
    # Create complex SED data (3 polarizations)
    sed_complex = np.zeros((len(freqs), len(k_points), 3), dtype=np.complex64)
    sed_complex[:, :, 0] = np.sqrt(intensity)  # Store in first polarization
    
    # Create k-vectors (these are 3D vectors for each k-point magnitude)
    # For a 1D path, k_vectors would be k_magnitudes * direction_vector
    k_vectors_3d = np.zeros((len(k_points), 3), dtype=np.float32)
    k_vectors_3d[:, 0] = k_points  # Assuming k_points are along x for this example
    
    return SED(sed=sed_complex, freqs=freqs, k_points=k_points, k_vectors=k_vectors_3d)

def main():
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create sample SED data
    sed_data = create_sample_sed_data()
    
    # Create 2D intensity plot
    print("Generating 2D intensity plot...")
    plotter_2d = SEDPlotter(
        sed_data,  # sed_obj (positional)
        '2d_intensity',  # plot_type (positional)
        str(output_dir / "sed_intensity_2d.png"),  # output_path (positional)
        title="Sample SED Intensity Plot",
        direction_label="[100] direction", # This will be combined with k-units
        max_freq=30.0,  # Show only up to 30 THz
        highlight_region={
            'k_point_target': np.pi,
            'freq_point_target': 15.0
        },
        log_intensity=False # Explicitly set, can be True
    )
    plotter_2d.generate_plot()
    
    # Create 1D slice at k = pi (Re-enabling this plot)
    print("Generating 1D slice plot at k=pi...")
    k_target_for_slice = np.pi
    k_index_for_slice = np.argmin(np.abs(sed_data.k_points - k_target_for_slice))
    plotter_1d_k = SEDPlotter(
        sed_data, # sed_obj (positional)
        '1d_slice', # plot_type (positional)
        str(output_dir / "sed_slice_k_pi.png"), # output_path (positional)
        title=f"SED Intensity at k ≈ {sed_data.k_points[k_index_for_slice]:.2f} (index {k_index_for_slice})",
        k_index=k_index_for_slice # Note: SEDPlotter expects k_index or freq_index
    )
    plotter_1d_k.generate_plot()

    # Create 1D slice at freq = 20 THz
    print("Generating 1D slice plot at freq=20 THz...")
    freq_target_for_slice = 20.0
    freq_index_for_slice = np.argmin(np.abs(sed_data.freqs - freq_target_for_slice))
    plotter_1d_f = SEDPlotter(
        sed_data, # sed_obj (positional)
        '1d_slice', # plot_type (positional)
        str(output_dir / "sed_slice_freq_20THz.png"), # output_path (positional)
        title=f"SED Intensity at freq ≈ {sed_data.freqs[freq_index_for_slice]:.2f} THz (index {freq_index_for_slice})",
        freq_index=freq_index_for_slice # Note: SEDPlotter expects k_index or freq_index
    )
    plotter_1d_f.generate_plot()
    
    # Create frequency slice plot at target_frequency = 15 THz
    print("Generating frequency slice plot at target_frequency=15 THz...")
    target_freq_for_slice = 15.0
    plotter_freq_slice = SEDPlotter(
        sed_data,  # sed_obj (positional)
        'frequency_slice',  # plot_type (positional)
        str(output_dir / f"sed_frequency_slice_{target_freq_for_slice:.0f}THz.png"),  # output_path (positional)
        title=f"SED Intensity vs k-points at target ω ≈ {target_freq_for_slice:.1f} THz",
        target_frequency=target_freq_for_slice,
        direction_label="[100] direction",
        log_intensity=True # Example with log intensity
    )
    plotter_freq_slice.generate_plot()

    # Example: frequency slice plot at another frequency, e.g. 35 THz (might be noisy or low signal)
    print("Generating frequency slice plot at target_frequency=35 THz...")
    target_freq_for_slice_2 = 35.0
    plotter_freq_slice_2 = SEDPlotter(
        sed_data,  # sed_obj (positional)
        'frequency_slice',  # plot_type (positional)
        str(output_dir / f"sed_frequency_slice_{target_freq_for_slice_2:.0f}THz_linear.png"),  # output_path (positional)
        title=f"SED Intensity vs k-points at target ω ≈ {target_freq_for_slice_2:.1f} THz (Linear)",
        target_frequency=target_freq_for_slice_2,
        direction_label="[100] direction",
        log_intensity=False # Example with linear intensity
    )
    plotter_freq_slice_2.generate_plot()
    
    print(f"Plots have been saved to {output_dir}")

if __name__ == "__main__":
    main() 