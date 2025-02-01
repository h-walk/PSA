#!/usr/bin/env python3
"""
Plot Spectral Displacement Data Slices from a .npz File with Optional Normalization and GIF Generation

This script loads a .npz file containing 'kx', 'ky', 'freq', and 'amp' arrays,
applies optional masks, extends the data to all four quadrants by symmetry,
slices the data along the frequency axis with Gaussian weighting (FWHM=1 THz),
generates heatmaps for each frequency slice, and optionally creates an animated GIF
from the generated heatmaps.

Usage:
    python plot_sd_slices.py --input data.npz --output_dir plots/
                             [--kx_min KX_MIN] [--kx_max KX_MAX]
                             [--ky_min KY_MIN] [--ky_max KY_MAX]
                             [--freq_min FREQ_MIN] [--freq_max FREQ_MAX]
                             [--bins_kx BINS_KX] [--bins_ky BINS_KY]
                             [--slice_step SLICE_STEP]
                             [--fwhm FWHM]
                             [--normalize]
                             [--gif]
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

# Import imageio only if GIF generation is requested
try:
    import imageio
except ImportError:
    imageio = None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot Spectral Displacement Data Slices from a .npz File with Optional Normalization and GIF Generation.'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the input .npz file containing kx, ky, freq, amp arrays.')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Directory to save the output heatmap plots and GIF.')

    # Optional masking arguments
    parser.add_argument('--kx_min', type=float, default=None, help='Minimum kx value to include.')
    parser.add_argument('--kx_max', type=float, default=None, help='Maximum kx value to include.')
    parser.add_argument('--ky_min', type=float, default=None, help='Minimum ky value to include.')
    parser.add_argument('--ky_max', type=float, default=None, help='Maximum ky value to include.')
    parser.add_argument('--freq_min', type=float, default=None, help='Minimum frequency to include.')
    parser.add_argument('--freq_max', type=float, default=None, help='Maximum frequency to include.')

    # Binning and slicing parameters
    parser.add_argument('--bins_kx', type=int, default=250, help='Number of bins for kx axis in heatmaps.')
    parser.add_argument('--bins_ky', type=int, default=250, help='Number of bins for ky axis in heatmaps.')
    parser.add_argument('--slice_step', type=float, default=0.5, help='Frequency step between slice centers (in THz).')
    parser.add_argument('--fwhm', type=float, default=1.0, help='Full Width at Half Maximum for Gaussian slicing (in THz).')

    # Normalization toggle
    parser.add_argument('--normalize', action='store_true',
                        help='Enable normalization and √ scaling of heatmap intensities across all frequency slices.')

    # GIF generation toggle
    parser.add_argument('--gif', action='store_true',
                        help='Generate an animated GIF from the generated heatmaps.')

    return parser.parse_args()


def load_npz(npz_path):
    if not Path(npz_path).is_file():
        print(f"Error: File '{npz_path}' does not exist.")
        sys.exit(1)

    data = np.load(npz_path)
    required_keys = {'kx', 'ky', 'freq', 'amp'}
    if not required_keys.issubset(data.files):
        print(f"Error: .npz file must contain the following arrays: {required_keys}")
        print(f"Found arrays: {data.files}")
        sys.exit(1)

    kx = data['kx']
    ky = data['ky']
    freq = data['freq']
    amp = data['amp']

    return kx, ky, freq, amp


def apply_masks(kx, ky, freq, amp, args):
    """
    Apply min and max masks to the data based on command-line arguments.
    """
    mask = np.ones_like(kx, dtype=bool)

    if args.kx_min is not None:
        mask &= (kx >= args.kx_min)
    if args.kx_max is not None:
        mask &= (kx <= args.kx_max)
    if args.ky_min is not None:
        mask &= (ky >= args.ky_min)
    if args.ky_max is not None:
        mask &= (ky <= args.ky_max)
    if args.freq_min is not None:
        mask &= (freq >= args.freq_min)
    if args.freq_max is not None:
        mask &= (freq <= args.freq_max)

    kx_filtered = kx[mask]
    ky_filtered = ky[mask]
    freq_filtered = freq[mask]
    amp_filtered = amp[mask]

    print(f"Total data points before masking: {len(kx)}")
    print(f"Data points after masking: {len(kx_filtered)}")

    return kx_filtered, ky_filtered, freq_filtered, amp_filtered


def apply_symmetry(kx, ky, freq, amp):
    """
    Extend the data to all four quadrants by reflecting across kx and ky axes.

    Args:
        kx (np.ndarray): Array of kx values (positive only).
        ky (np.ndarray): Array of ky values (positive only).
        freq (np.ndarray): Array of frequency values (positive only).
        amp (np.ndarray): Array of amplitude values.

    Returns:
        kx_full, ky_full, freq_full, amp_full (np.ndarray): Symmetrically extended arrays.
    """
    # Original data
    kx_full = kx.copy()
    ky_full = ky.copy()
    freq_full = freq.copy()
    amp_full = amp.copy()

    # Reflect across kx axis
    mask_kx_nonzero = kx != 0
    kx_neg = -kx[mask_kx_nonzero]
    ky_neg = ky[mask_kx_nonzero]
    freq_neg = freq[mask_kx_nonzero]
    amp_neg = amp[mask_kx_nonzero]

    kx_full = np.concatenate([kx_full, kx_neg])
    ky_full = np.concatenate([ky_full, ky_neg])
    freq_full = np.concatenate([freq_full, freq_neg])
    amp_full = np.concatenate([amp_full, amp_neg])

    # Reflect across ky axis
    mask_ky_nonzero = ky != 0
    kx_neg_ky = kx[mask_ky_nonzero]
    ky_neg_ky = -ky[mask_ky_nonzero]
    freq_neg_ky = freq[mask_ky_nonzero]
    amp_neg_ky = amp[mask_ky_nonzero]

    kx_full = np.concatenate([kx_full, kx_neg_ky])
    ky_full = np.concatenate([ky_full, ky_neg_ky])
    freq_full = np.concatenate([freq_full, freq_neg_ky])
    amp_full = np.concatenate([amp_full, amp_neg_ky])

    # Reflect across both axes
    # To avoid duplicating points where kx or ky are zero, exclude those
    mask_both_nonzero = (kx != 0) & (ky != 0)
    kx_neg_both = -kx[mask_both_nonzero]
    ky_neg_both = -ky[mask_both_nonzero]
    freq_neg_both = freq[mask_both_nonzero]
    amp_neg_both = amp[mask_both_nonzero]

    kx_full = np.concatenate([kx_full, kx_neg_both])
    ky_full = np.concatenate([ky_full, ky_neg_both])
    freq_full = np.concatenate([freq_full, freq_neg_both])
    amp_full = np.concatenate([amp_full, amp_neg_both])

    return kx_full, ky_full, freq_full, amp_full


def gaussian_weight(w, w0, sigma):
    """
    Compute Gaussian weights for frequency slicing.

    Args:
        w: Array of frequency values.
        w0: Center frequency of the slice.
        sigma: Standard deviation of the Gaussian.

    Returns:
        Gaussian weights.
    """
    return np.exp(-0.5 * ((w - w0) / sigma)**2)


def create_heatmap(kx, ky, amp, bins_kx, bins_ky):
    """
    Create a 2D histogram (heatmap) of kx vs ky, weighted by amp.

    Args:
        kx (np.ndarray): Array of kx values.
        ky (np.ndarray): Array of ky values.
        amp (np.ndarray): Array of amplitude values.
        bins_kx (int): Number of bins for kx axis.
        bins_ky (int): Number of bins for ky axis.

    Returns:
        heatmap, x_edges, y_edges (np.ndarray): 2D heatmap and bin edges.
    """
    heatmap, x_edges, y_edges, _ = binned_statistic_2d(
        kx, ky, amp, statistic='sum', bins=[bins_kx, bins_ky], expand_binnumbers=False
    )
    return heatmap, x_edges, y_edges


def plot_heatmap(heatmap, x_edges, y_edges, w0, output_path, normalize=False, global_max=None):
    """
    Plot and save the heatmap with optional normalization and √ scaling.

    Args:
        heatmap (np.ndarray): 2D array representing the heatmap.
        x_edges (np.ndarray): Bin edges for kx.
        y_edges (np.ndarray): Bin edges for ky.
        w0 (float): Center frequency of the slice.
        output_path (str or Path): Path to save the heatmap image.
        normalize (bool): Whether to normalize and apply √ scaling.
        global_max (float, optional): The maximum heatmap value across all slices for normalization.
    """
    plt.figure(figsize=(8, 6))
    # Define extent to map the bin edges to the axes
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    if normalize and global_max is not None:
        # Normalize the heatmap to range [0, 1] based on global max
        heatmap_normalized = heatmap / global_max

        # Apply square root scaling to enhance lower intensities
        heatmap_scaled = np.sqrt(np.sqrt(np.sqrt(heatmap_normalized)))

        # Handle cases where heatmap has zeros or very small values
        # To prevent plotting issues, set NaN where scaled heatmap is zero
        heatmap_safe = np.where(heatmap_scaled > 0, heatmap_scaled, np.nan)

        # Plot using imshow with linear normalization from 0 to 1
        img = plt.imshow(heatmap_safe.T, origin='lower', extent=extent, aspect='equal',
                         cmap='inferno', norm=plt.Normalize(vmin=0, vmax=1))

        # Add colorbar with label
        cbar = plt.colorbar(img, label='Normalized √ Scaled Amplitude (0-1)')

        # Set title
        plt.title(f'Frequency Slice: {w0:.2f} THz (Normalized)')
    else:
        # Handle cases where heatmap has zeros or very small values
        heatmap_safe = np.where(heatmap > 0, heatmap, np.nan)

        # Plot using imshow with logarithmic color scale
        img = plt.imshow(heatmap_safe.T, origin='lower', extent=extent, aspect='equal',
                         cmap='inferno', norm=plt.matplotlib.colors.LogNorm())

        # Add colorbar with label
        cbar = plt.colorbar(img, label='Amplitude (arb. units)')

        # Set title
        plt.title(f'Frequency Slice: {w0:.2f} THz (Raw)')

    # Set labels
    plt.xlabel(r'$k_x$ (2$\pi$/Å)')
    plt.ylabel(r'$k_y$ (2$\pi$/Å)')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_gif(image_paths, gif_path, duration=0.5):
    """
    Create an animated GIF from a list of image paths.

    Args:
        image_paths (list): List of file paths to the heatmap images.
        gif_path (str or Path): Path to save the animated GIF.
        duration (float): Duration between frames in seconds.
    """
    if imageio is None:
        print("Error: imageio library is not installed. Install it using 'pip install imageio'")
        sys.exit(1)

    images = []
    for filename in image_paths:
        images.append(imageio.imread(filename))

    # Save the GIF with loop=0 for infinite looping
    imageio.mimsave(gif_path, images, duration=duration, loop=0)
    print(f"Animated GIF saved as '{gif_path}'")


def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    kx, ky, freq, amp = load_npz(args.input)

    # Apply masks
    kx_filtered, ky_filtered, freq_filtered, amp_filtered = apply_masks(kx, ky, freq, amp, args)

    if len(kx_filtered) == 0:
        print("No data points to plot after applying masks. Exiting.")
        sys.exit(1)

    # Apply symmetry to span all four quadrants
    kx_full, ky_full, freq_full, amp_full = apply_symmetry(kx_filtered, ky_filtered, freq_filtered, amp_filtered)
    print(f"Data points after symmetry transformation: {len(kx_full)}")

    # Define Gaussian slicing parameters
    fwhm = args.fwhm  # in THz
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    print(f"Gaussian slicing parameters: FWHM={fwhm} THz, sigma={sigma:.3f} THz")

    # Define slice centers
    w_min = freq_full.min()
    w_max = freq_full.max()
    slice_step = args.slice_step
    slice_centers = np.arange(w_min, w_max + slice_step, slice_step)
    print(f"Creating {len(slice_centers)} frequency slices from {w_min:.2f} to {w_max:.2f} THz with step {slice_step} THz.")

    # Define kx and ky bin counts
    bins_kx = args.bins_kx
    bins_ky = args.bins_ky

    # Initialize list to store image paths for GIF
    gif_image_paths = []

    # Check if normalization is enabled
    if args.normalize:
        # First Pass: Find the global maximum heatmap value across all slices
        global_max = 0
        print("First Pass: Determining global maximum heatmap value for normalization...")
        for w0 in tqdm(slice_centers, desc="Processing frequency slices (First Pass)", unit="slice"):
            weights = gaussian_weight(freq_full, w0, sigma)
            amp_weighted = amp_full * weights

            # Skip slices with negligible weights to save computation
            if amp_weighted.sum() < 1e-3:
                continue

            # Create heatmap
            heatmap, _, _ = create_heatmap(kx_full, ky_full, amp_weighted, bins_kx, bins_ky)

            # Update global maximum if current heatmap has a higher value
            current_max = np.nanmax(heatmap)
            if current_max > global_max:
                global_max = current_max

        if global_max == 0:
            print("All slices have negligible amplitudes. Exiting.")
            sys.exit(1)

        print(f"Global maximum heatmap value across all slices: {global_max:.6f}")

    # Second Pass: Generate and save heatmaps
    if args.normalize:
        print("Second Pass: Generating and saving normalized √ scaled heatmaps...")
    else:
        print("Generating and saving raw heatmaps...")

    for w0 in tqdm(slice_centers, desc="Processing frequency slices (Second Pass)", unit="slice"):
        weights = gaussian_weight(freq_full, w0, sigma)
        amp_weighted = amp_full * weights

        # Skip slices with negligible weights
        if amp_weighted.sum() < 1e-3:
            continue

        # Create heatmap
        heatmap, x_edges, y_edges = create_heatmap(kx_full, ky_full, amp_weighted, bins_kx, bins_ky)

        # Define output path
        output_path = output_dir / f"heatmap_w_{w0:.2f}_THz.png"

        # Plot and save the heatmap
        if args.normalize:
            plot_heatmap(heatmap, x_edges, y_edges, w0, output_path, normalize=True, global_max=global_max)
        else:
            plot_heatmap(heatmap, x_edges, y_edges, w0, output_path, normalize=False)

        # Append image path for GIF
        if args.gif:
            gif_image_paths.append(str(output_path))

    # Generate GIF if requested
    if args.gif:
        if len(gif_image_paths) == 0:
            print("No heatmap images available to create GIF. Skipping GIF generation.")
        else:
            gif_path = output_dir / "animation.gif"
            print("Generating animated GIF...")
            create_gif(gif_image_paths, gif_path, duration=0.5)  # duration is frame delay in seconds

    print(f"All heatmaps have been saved to '{output_dir.resolve()}'.")
    if args.gif and len(gif_image_paths) > 0:
        print(f"Animated GIF has been saved to '{gif_path.resolve()}'.")
    

if __name__ == "__main__":
    main()
