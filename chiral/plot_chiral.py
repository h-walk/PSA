#!/usr/bin/env python3
"""
Plot Chiral (Relative Phase) Heatmaps

This script loads a chiral.npz file (output by SDA.py) that contains,
for each direction index i, keys:
    - dir_{i}_angle      (scalar, degrees)
    - dir_{i}_k_points   (1D array, length n_k)
    - dir_{i}_freqs      (1D array, length n_f)
    - dir_{i}_{pair}     (2D array of shape (n_f, n_k) with relative phase in radians)
It then builds a grid for each segment, concatenates them, tiles the data
(using the specified angle_range, e.g. [0,120] yields 3 copies for a full 360°),
applies frequency masks, and for each frequency slice (using a simple window)
bins the (kx,ky) data to compute the circular mean phase in each bin.
The heatmaps are plotted with the HSV colormap (vmin=-π, vmax=π).
Optionally an animated GIF is produced.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

try:
    import imageio
except ImportError:
    imageio = None

# -------------------- Argument Parsing --------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot chiral (relative phase) heatmaps with symmetry tiling.'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the chiral NPZ file.')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Directory to save the output heatmap images and GIF.')

    # These are the lines you need to add or confirm exist:
    parser.add_argument('--kx_min', type=float, default=None, help='Minimum kx value to include.')
    parser.add_argument('--kx_max', type=float, default=None, help='Maximum kx value to include.')
    parser.add_argument('--ky_min', type=float, default=None, help='Minimum ky value to include.')
    parser.add_argument('--ky_max', type=float, default=None, help='Maximum ky value to include.')
    parser.add_argument('--freq_min', type=float, default=None, help='Minimum frequency to include.')
    parser.add_argument('--freq_max', type=float, default=None, help='Maximum frequency to include.')

    # The rest of your arguments
    parser.add_argument('--angle_range', type=float, nargs=2, default=[0, 120],
                        help='Angle range (in degrees) of the input data (e.g. 0 120).')
    parser.add_argument('--bins_kx', type=int, default=120, help='Number of bins for kx axis.')
    parser.add_argument('--bins_ky', type=int, default=120, help='Number of bins for ky axis.')
    parser.add_argument('--slice_step', type=float, default=0.5,
                        help='Frequency step between slice centers (THz).')
    parser.add_argument('--fwhm', type=float, default=1.0,
                        help='Full Width at Half Maximum for frequency slicing (THz).')
    parser.add_argument('--gif', action='store_true',
                        help='Generate an animated GIF from the heatmaps.')
    parser.add_argument('--pair', type=str, choices=['phase_0_1', 'phase_0_2', 'phase_1_2'],
                        default='phase_0_1',
                        help='Polarization pair to use (default: phase_0_1).')

    return parser.parse_args()
# -------------------- Data Loading --------------------
def load_chiral_segments(npz_path, pair="phase_0_1"):
    """
    Load all segments from the NPZ file.
    For each segment i (with keys "dir_i_angle", etc.):

      - Reads the angle (in degrees), converts to radians.
      - Reads k_points (1D), freqs (1D), and the phase array (2D, shape (n_f, n_k)).
      - Checks that phase.shape equals (len(freqs), len(k_points)).
      - Constructs:
            kx_i = tile( k_points*cos(angle_rad), n_f )
            ky_i = tile( k_points*sin(angle_rad), n_f )
            freq_i = repeat( freqs, n_k )
            phase_i = phase.flatten()   (relative phase as stored)
    Returns concatenated 1D arrays: kx, ky, freq, phase.
    """
    data = np.load(npz_path)
    indices = []
    for key in data.files:
        if key.startswith("dir_") and key.endswith("_angle"):
            try:
                indices.append(int(key.split('_')[1]))
            except Exception:
                continue
    indices = sorted(set(indices))
    if not indices:
        print("Error: No chiral directional data found.")
        sys.exit(1)
    kx_list, ky_list, freq_list, phase_list = [], [], [], []
    for i in indices:
        try:
            angle_deg = data[f"dir_{i}_angle"]
            angle_rad = np.deg2rad(angle_deg)
            k_points = np.array(data[f"dir_{i}_k_points"]).flatten()  # shape (n_k,)
            freqs = np.array(data[f"dir_{i}_freqs"]).flatten()          # shape (n_f,)
            phase_arr = np.array(data[f"dir_{i}_{pair}"])               # shape (n_f, n_k)
        except KeyError as e:
            print(f"Warning: Missing key in segment {i}: {e}. Skipping.")
            continue

        if phase_arr.shape != (len(freqs), len(k_points)):
            print(f"Warning: Mismatch in segment {i}: k_points {len(k_points)}, "
                  f"freqs {len(freqs)}, phase shape {phase_arr.shape}. Skipping.")
            continue

        kx_i = np.tile(k_points * np.cos(angle_rad), len(freqs))
        ky_i = np.tile(k_points * np.sin(angle_rad), len(freqs))
        freq_i = np.repeat(freqs, len(k_points))
        phase_i = phase_arr.flatten()
        kx_list.append(kx_i)
        ky_list.append(ky_i)
        freq_list.append(freq_i)
        phase_list.append(phase_i)
    if not kx_list:
        print("Error: No valid segments loaded.")
        sys.exit(1)
    kx_all = np.concatenate(kx_list)
    ky_all = np.concatenate(ky_list)
    freq_all = np.concatenate(freq_list)
    phase_all = np.concatenate(phase_list)
    return kx_all, ky_all, freq_all, phase_all


# -------------------- Symmetry Tiling --------------------
def apply_symmetry(kx, ky, freq, phase, angle_range):
    """
    Tile the input data by rotating it.
    Suppose the input covers an angular span of angle_range = [angle_min, angle_max].
    Then copies = 360 / (angle_max - angle_min). Each copy is rotated by multiples
    of (angle_max - angle_min). (freq and phase remain unchanged.)
    """
    angle_min, angle_max = angle_range
    width = angle_max - angle_min
    if width <= 0:
        raise ValueError("Invalid angle_range: second value must be greater than the first.")
    copies = int(round(360 / width))
    new_kx, new_ky = [], []
    for i in range(copies):
        theta = np.deg2rad(i * width)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        new_kx.append(cos_t * kx - sin_t * ky)
        new_ky.append(sin_t * kx + cos_t * ky)
    return (np.concatenate(new_kx), np.concatenate(new_ky),
            np.tile(freq, copies), np.tile(phase, copies))


# -------------------- Masking --------------------
def apply_masks(kx, ky, freq, phase, args):
    mask = np.ones(kx.shape, dtype=bool)
    if args.kx_min is not None: mask &= (kx >= args.kx_min)
    if args.kx_max is not None: mask &= (kx <= args.kx_max)
    if args.ky_min is not None: mask &= (ky >= args.ky_min)
    if args.ky_max is not None: mask &= (ky <= args.ky_max)
    if args.freq_min is not None: mask &= (freq >= args.freq_min)
    if args.freq_max is not None: mask &= (freq <= args.freq_max)
    print(f"Points before masking: {len(kx)}; after: {np.sum(mask)}")
    return kx[mask], ky[mask], freq[mask], phase[mask]


# -------------------- Circular Mean Statistic --------------------
def circular_mean(arr):
    if len(arr) == 0:
        return np.nan
    return np.angle(np.mean(np.exp(1j * arr)))


def create_heatmap(kx, ky, phase, bins_kx, bins_ky):
    """
    Bin the (kx,ky) data and compute the circular mean of phase values in each bin.
    """
    heatmap, x_edges, y_edges, _ = binned_statistic_2d(
        kx, ky, phase, statistic=circular_mean, bins=[bins_kx, bins_ky], expand_binnumbers=False
    )
    return heatmap, x_edges, y_edges


# -------------------- Frequency Slicing --------------------
def gaussian_weight(w, w0, sigma):
    return np.exp(-0.5 * ((w - w0) / sigma)**2)


# -------------------- Plotting Heatmaps --------------------
def plot_heatmap(heatmap, x_edges, y_edges, w0, output_path):
    plt.figure(figsize=(8,6))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    img = plt.imshow(heatmap.T, origin='lower', extent=extent, aspect='equal',
                     cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(img, label='Relative Phase (rad)')
    plt.title(f'Freq Slice: {w0:.2f} THz')
    plt.xlabel(r'$k_x$ (2$\pi$/Å)')
    plt.ylabel(r'$k_y$ (2$\pi$/Å)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_gif(image_paths, gif_path, duration=0.5):
    if imageio is None:
        print("Error: imageio not installed. (Try 'pip install imageio')")
        sys.exit(1)
    images = [imageio.imread(fn) for fn in image_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=0)
    print(f"Animated GIF saved as '{gif_path}'")


# -------------------- Main Routine --------------------
def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load chiral segments.
    print("Loading chiral data from NPZ...")
    kx, ky, freq, phase = load_chiral_segments(args.input, pair=args.pair)
    print(f"Loaded {len(kx)} points from chiral segments.")

    # Tile data to cover full circle.
    kx, ky, freq, phase = apply_symmetry(kx, ky, freq, phase, args.angle_range)
    print(f"Data points after tiling: {len(kx)}")

    # Apply masks.
    kx, ky, freq, phase = apply_masks(kx, ky, freq, phase, args)
    if kx.size == 0:
        print("No points left after masking. Exiting.")
        sys.exit(1)

    # Set Gaussian slicing parameters.
    fwhm = args.fwhm
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    print(f"Gaussian slicing: FWHM={fwhm:.2f} THz, sigma={sigma:.3f} THz")
    # Use the masked frequency array to determine slice range.
    w_min, w_max = freq.min(), freq.max()
    slice_step = args.slice_step
    slice_centers = np.arange(w_min, w_max + slice_step, slice_step)
    print(f"{len(slice_centers)} frequency slices from {w_min:.2f} to {w_max:.2f} THz.")

    bins_kx = args.bins_kx
    bins_ky = args.bins_ky

    gif_image_paths = []
    # For each frequency slice, select points within a window.
    # Here we use a simple window: points with freq in [w0 - slice_step/2, w0 + slice_step/2].
    for w0 in tqdm(slice_centers, desc="Frequency slices", unit="slice"):
        mask = (freq >= w0 - slice_step/2) & (freq < w0 + slice_step/2)
        if np.sum(mask) < 10:
            continue
        kx_slice = kx[mask]
        ky_slice = ky[mask]
        phase_slice = phase[mask]
        # Compute heatmap using circular mean.
        heatmap, x_edges, y_edges = create_heatmap(kx_slice, ky_slice, phase_slice, bins_kx, bins_ky)
        output_path = output_dir / f"heatmap_w_{w0:.2f}_THz.png"
        plot_heatmap(heatmap, x_edges, y_edges, w0, output_path)
        gif_image_paths.append(str(output_path))

    if args.gif and gif_image_paths:
        gif_path = output_dir / "animation.gif"
        create_gif(gif_image_paths, gif_path, duration=0.5)
    
    print(f"Heatmaps saved in '{output_dir.resolve()}'.")


if __name__ == "__main__":
    main()
