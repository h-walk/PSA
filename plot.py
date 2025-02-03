#!/usr/bin/env python3
"""
Plot Spectral Displacement Data Slices from a .npz File with Optional Normalization, 
GIF Generation, and adjustable Brillouin zone tiling based on an input angle range.

Usage:
    python plot_sd_slices.py --input data.npz --output_dir plots/ [--angle_range 0 90]
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

try:
    import imageio
except ImportError:
    imageio = None

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot SD data slices with adjustable BZ tiling via angle_range.'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the input .npz file containing kx, ky, freq, amp arrays.')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Directory to save the output heatmap plots and GIF.')
    parser.add_argument('--angle_range', type=float, nargs=2, default=[0, 90],
                        help='Angle range (in degrees) of the input data. For example, [0 120] yields 3-fold symmetry, [0 90] yields 4-fold.')
    parser.add_argument('--kx_min', type=float, default=None, help='Minimum kx value to include.')
    parser.add_argument('--kx_max', type=float, default=None, help='Maximum kx value to include.')
    parser.add_argument('--ky_min', type=float, default=None, help='Minimum ky value to include.')
    parser.add_argument('--ky_max', type=float, default=None, help='Maximum ky value to include.')
    parser.add_argument('--freq_min', type=float, default=None, help='Minimum frequency to include.')
    parser.add_argument('--freq_max', type=float, default=None, help='Maximum frequency to include.')
    parser.add_argument('--bins_kx', type=int, default=120, help='Number of bins for kx axis in heatmaps.')
    parser.add_argument('--bins_ky', type=int, default=120, help='Number of bins for ky axis in heatmaps.')
    parser.add_argument('--slice_step', type=float, default=0.5, help='Frequency step between slice centers (in THz).')
    parser.add_argument('--fwhm', type=float, default=1.0, help='Full Width at Half Maximum for Gaussian slicing (in THz).')
    parser.add_argument('--normalize', action='store_true',
                        help='Enable normalization and √ scaling of heatmap intensities across all frequency slices.')
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
        print(f"Error: .npz file must contain arrays: {required_keys}. Found: {data.files}")
        sys.exit(1)
    return data['kx'], data['ky'], data['freq'], data['amp']

def apply_masks(kx, ky, freq, amp, args):
    mask = np.ones_like(kx, dtype=bool)
    if args.kx_min is not None: mask &= (kx >= args.kx_min)
    if args.kx_max is not None: mask &= (kx <= args.kx_max)
    if args.ky_min is not None: mask &= (ky >= args.ky_min)
    if args.ky_max is not None: mask &= (ky <= args.ky_max)
    if args.freq_min is not None: mask &= (freq >= args.freq_min)
    if args.freq_max is not None: mask &= (freq <= args.freq_max)
    kx_f = kx[mask]
    ky_f = ky[mask]
    freq_f = freq[mask]
    amp_f = amp[mask]
    print(f"Total points before masking: {len(kx)}; after: {len(kx_f)}")
    return kx_f, ky_f, freq_f, amp_f

def apply_symmetry(kx, ky, freq, amp, angle_range):
    """
    Tile the input data by rotating it according to the specified angle_range.
    The input data is assumed to cover the angular span defined by angle_range.
    The number of copies is 360 divided by (angle_max - angle_min).
    """
    angle_min, angle_max = angle_range
    width = angle_max - angle_min
    if width <= 0:
        raise ValueError("Invalid angle_range: second value must be greater than the first.")
    copies = int(round(360 / width))
    new_kx, new_ky, new_freq, new_amp = [], [], [], []
    for i in range(copies):
        theta_deg = i * width
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate each (kx, ky) point
        kx_rot = cos_t * kx - sin_t * ky
        ky_rot = sin_t * kx + cos_t * ky
        new_kx.append(kx_rot)
        new_ky.append(ky_rot)
        new_freq.append(freq)  # unchanged
        new_amp.append(amp)    # unchanged
    return (np.concatenate(new_kx), np.concatenate(new_ky),
            np.concatenate(new_freq), np.concatenate(new_amp))

def gaussian_weight(w, w0, sigma):
    return np.exp(-0.5 * ((w - w0) / sigma)**2)

def create_heatmap(kx, ky, amp, bins_kx, bins_ky):
    heatmap, x_edges, y_edges, _ = binned_statistic_2d(
        kx, ky, amp, statistic='sum', bins=[bins_kx, bins_ky], expand_binnumbers=False
    )
    return heatmap, x_edges, y_edges

def plot_heatmap(heatmap, x_edges, y_edges, w0, output_path, normalize=False, global_max=None):
    plt.figure(figsize=(8, 6))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    if normalize and global_max is not None:
        heatmap_norm = heatmap / global_max
        heatmap_scaled = np.sqrt(np.sqrt(np.sqrt(heatmap_norm)))
        heatmap_safe = np.where(heatmap_scaled > 0, heatmap_scaled, np.nan)
        img = plt.imshow(heatmap_safe.T, origin='lower', extent=extent, aspect='equal',
                         cmap='inferno', norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(img, label='Normalized √ Scaled Amplitude (0-1)')
        plt.title(f'Freq Slice: {w0:.2f} THz (Normalized)')
    else:
        heatmap_safe = np.where(heatmap > 0, heatmap, np.nan)
        img = plt.imshow(heatmap_safe.T, origin='lower', extent=extent, aspect='equal',
                         cmap='inferno', norm=plt.matplotlib.colors.LogNorm())
        plt.colorbar(img, label='Amplitude (arb. units)')
        plt.title(f'Freq Slice: {w0:.2f} THz (Raw)')
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

def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kx, ky, freq, amp = load_npz(args.input)
    kx_f, ky_f, freq_f, amp_f = apply_masks(kx, ky, freq, amp, args)
    if len(kx_f) == 0:
        print("No points to plot after masking. Exiting.")
        sys.exit(1)
    # Tile the data using the specified angle range
    kx_full, ky_full, freq_full, amp_full = apply_symmetry(kx_f, ky_f, freq_f, amp_f, args.angle_range)
    print(f"Data points after tiling: {len(kx_full)}")
    fwhm = args.fwhm
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    print(f"Gaussian slicing: FWHM={fwhm} THz, sigma={sigma:.3f} THz")
    w_min, w_max = freq_full.min(), freq_full.max()
    slice_step = args.slice_step
    slice_centers = np.arange(w_min, w_max + slice_step, slice_step)
    print(f"{len(slice_centers)} frequency slices from {w_min:.2f} to {w_max:.2f} THz (step {slice_step} THz).")
    bins_kx = args.bins_kx
    bins_ky = args.bins_ky
    gif_image_paths = []
    global_max = 0
    if args.normalize:
        print("First Pass: Determining global max for normalization...")
        for w0 in tqdm(slice_centers, desc="Freq slices (Pass 1)", unit="slice"):
            weights = gaussian_weight(freq_full, w0, sigma)
            amp_w = amp_full * weights
            if amp_w.sum() < 1e-3:
                continue
            heatmap, _, _ = create_heatmap(kx_full, ky_full, amp_w, bins_kx, bins_ky)
            current_max = np.nanmax(heatmap)
            if current_max > global_max:
                global_max = current_max
        if global_max == 0:
            print("Negligible amplitudes in all slices. Exiting.")
            sys.exit(1)
        print(f"Global maximum heatmap value: {global_max:.6f}")
    if args.normalize:
        print("Second Pass: Generating normalized √ scaled heatmaps...")
    else:
        print("Generating raw heatmaps...")
    for w0 in tqdm(slice_centers, desc="Freq slices (Pass 2)", unit="slice"):
        weights = gaussian_weight(freq_full, w0, sigma)
        amp_w = amp_full * weights
        if amp_w.sum() < 1e-3:
            continue
        heatmap, x_edges, y_edges = create_heatmap(kx_full, ky_full, amp_w, bins_kx, bins_ky)
        output_path = output_dir / f"heatmap_w_{w0:.2f}_THz.png"
        if args.normalize:
            plot_heatmap(heatmap, x_edges, y_edges, w0, output_path, normalize=True, global_max=global_max)
        else:
            plot_heatmap(heatmap, x_edges, y_edges, w0, output_path, normalize=False)
        if args.gif:
            gif_image_paths.append(str(output_path))
    if args.gif:
        if not gif_image_paths:
            print("No images for GIF creation. Skipping GIF.")
        else:
            gif_path = output_dir / "animation.gif"
            print("Generating GIF...")
            create_gif(gif_image_paths, gif_path, duration=0.5)
    print(f"Heatmaps saved in '{output_dir.resolve()}'.")
    if args.gif and gif_image_paths:
        print(f"Animated GIF saved as '{gif_path.resolve()}'.")

if __name__ == "__main__":
    main()
