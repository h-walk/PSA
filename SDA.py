"""
Spectral Displacement (SD) Analysis Tool

This script performs SD analysis on molecular dynamics trajectories using the OVITO library.
It computes the SD, applies filtering, reconstructs atomic displacements, and outputs
the results both as plots and a LAMMPS-compatible trajectory file.

Dependencies:
- numpy
- ovito
- matplotlib
- tqdm
- yaml
"""

import numpy as np
import os
from pathlib import Path
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Union
import logging
import argparse
from tqdm import tqdm
import yaml

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

# Configure logging to display only essential information
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings to keep the output clean
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
warnings.filterwarnings('ignore', category=np.ComplexWarning)

from ovito.io import import_file

@dataclass
class Box:
    """
    Represents the simulation box with lengths, tilts, and the full cell matrix.
    """
    lengths: np.ndarray  # [lx, ly, lz]
    tilts: np.ndarray    # [xy, xz, yz]
    matrix: np.ndarray   # full 3x3 cell matrix

    @classmethod
    def from_ovito(cls, cell) -> 'Box':
        """
        Constructs a Box instance from an OVITO cell object.
        """
        matrix = cell.matrix.copy().astype(np.float32)
        lengths = np.array([
            matrix[0,0],
            matrix[1,1],
            matrix[2,2]
        ], dtype=np.float32)
        tilts = np.array([
            matrix[0,1],
            matrix[0,2],
            matrix[1,2]
        ], dtype=np.float32)
        return cls(lengths=lengths, tilts=tilts, matrix=matrix)
        
    def to_dict(self) -> Dict:
        """
        Converts the Box instance to a dictionary for saving.
        """
        return {
            'lengths': self.lengths,
            'tilts': self.tilts,
            'matrix': self.matrix
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Box':
        """
        Constructs a Box instance from a dictionary.
        """
        return cls(
            lengths=np.array(data['lengths'], dtype=np.float32),
            tilts=np.array(data['tilts'], dtype=np.float32),
            matrix=np.array(data['matrix'], dtype=np.float32)
        )

@dataclass
class Trajectory:
    """
    Stores trajectory data including positions, atom types, timesteps, and simulation box.
    """
    positions: np.ndarray  # Shape: (frames, atoms, 3)
    types: np.ndarray      # Shape: (atoms,)
    timesteps: np.ndarray  # Shape: (frames,)
    box: Box
    
    def __post_init__(self):
        """
        Validates the dimensions of the trajectory data.
        """
        if len(self.positions.shape) != 3:
            raise ValueError("Positions must be a 3D array (frames, atoms, xyz)")
        if len(self.types.shape) != 1:
            raise ValueError("Types must be a 1D array")
        if len(self.timesteps.shape) != 1:
            raise ValueError("Timesteps must be a 1D array")
    
    @property
    def n_frames(self) -> int:
        """Returns the number of frames in the trajectory."""
        return len(self.timesteps)
    
    @property
    def n_atoms(self) -> int:
        """Returns the number of atoms in the trajectory."""
        return len(self.types)

class TrajectoryLoader:
    """
    Handles loading and saving of trajectory data, utilizing OVITO for file parsing.
    """
    def __init__(self, filename: str, dt: float = 1.0):
        """
        Initializes the loader with the trajectory file path and timestep.
        
        Args:
            filename (str): Path to the trajectory file.
            dt (float): Time interval between frames (default: 1.0).
        """
        self.filepath = Path(filename)
        self.dt = dt
    
    def _get_save_filenames(self) -> Dict[str, str]:
        """
        Generates filenames for saving processed data based on the trajectory filename.
        """
        prefix = self.filepath.stem
        return {
            'pos': f"{prefix}_pos.npy",
            'types': f"{prefix}_types.npy",
            'time': f"{prefix}_time.npy",
            'box': f"{prefix}_box.npz"
        }
    
    def _load_saved(self) -> Optional[Trajectory]:
        """
        Attempts to load previously saved processed data to avoid reprocessing.
        
        Returns:
            Optional[Trajectory]: Loaded trajectory if available, else None.
        """
        filenames = self._get_save_filenames()
        if all(Path(f).exists() for f in filenames.values()):
            logger.info("Loading previously saved processed data...")
            try:
                box_data = dict(np.load(filenames['box'], allow_pickle=True))
                return Trajectory(
                    positions=np.load(filenames['pos'], allow_pickle=True).astype(np.float32),
                    types=np.load(filenames['types'], allow_pickle=True),
                    timesteps=np.load(filenames['time'], allow_pickle=True).astype(np.float32),
                    box=Box.from_dict(box_data)
                )
            except Exception as e:
                logger.warning(f"Failed to load saved files: {e}")
                return None
        return None
    
    def _save_processed(self, traj: Trajectory) -> None:
        """
        Saves the processed trajectory data to disk for future use.
        
        Args:
            traj (Trajectory): The trajectory data to save.
        """
        filenames = self._get_save_filenames()
        np.save(filenames['pos'], traj.positions)
        np.save(filenames['types'], traj.types)
        np.save(filenames['time'], traj.timesteps)
        np.savez(filenames['box'], **traj.box.to_dict())
        logger.info("Processed data saved for future use.")
    
    def load(self, reload: bool = False) -> Trajectory:
        """
        Loads the trajectory data, either from saved files or by parsing the original file.
        
        Args:
            reload (bool): If True, forces reloading from the original file.

        Returns:
            Trajectory: The loaded trajectory data.
        """
        if not reload:
            saved = self._load_saved()
            if saved is not None:
                return saved
        
        logger.info("Parsing trajectory from the original file...")
        pipeline = import_file(str(self.filepath))
        n_frames = pipeline.source.num_frames
        frame0 = pipeline.compute(0)
        n_atoms = len(frame0.particles.positions)
        
        # Initialize arrays to store trajectory data
        positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        types = frame0.particles.particle_types.array
        timesteps = np.arange(n_frames, dtype=np.float32) * self.dt
        box = Box.from_ovito(frame0.cell)
        
        # Read positions for each frame
        for i in tqdm(range(n_frames), desc="Reading frames", ncols=80):
            frame = pipeline.compute(i)
            positions[i] = frame.particles.positions.array.astype(np.float32)
        
        traj = Trajectory(positions, types, timesteps, box)
        self._save_processed(traj)
        return traj

class SDCalculator:
    """
    Calculates the Spectral Displacement (SD) from trajectory data.
    """
    def __init__(self, traj: Trajectory, nx: int, ny: int, nz: int):
        """
        Initializes the calculator with trajectory data and system size.
        
        Args:
            traj (Trajectory): The trajectory data.
            nx (int): Number of grid points in the x-direction.
            ny (int): Number of grid points in the y-direction.
            nz (int): Number of grid points in the z-direction.
        """
        self.traj = traj
        self.system_size = np.array([nx, ny, nz], dtype=np.int32)
        
        # Compute reciprocal lattice vectors
        cell_mat = self.traj.box.matrix.astype(np.float32)
        self.a1 = cell_mat[:,0] / float(nx)
        self.a2 = cell_mat[:,1] / float(ny)
        self.a3 = cell_mat[:,2] / float(nz)
        
        volume = np.dot(self.a1, np.cross(self.a2, self.a3))
        b1 = 2 * np.pi * np.cross(self.a2, self.a3) / volume
        b2 = 2 * np.pi * np.cross(self.a3, self.a1) / volume
        b3 = 2 * np.pi * np.cross(self.a1, self.a2) / volume
        self.recip_vectors = np.vstack([b1, b2, b3]).astype(np.float32)
        
        logger.info("Reciprocal lattice vectors (2π/Å):\n{}".format(self.recip_vectors))
        logger.info("Reciprocal lattice vector magnitudes (2π/Å): {}".format(
            [f"{np.linalg.norm(b):.3f}" for b in self.recip_vectors]
        ))
    
    def get_k_path(self, direction: Union[str, List[float]], bz_coverage: float, n_k: int, 
                   lattice_parameter: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a path of k-points along a specified direction for SD calculation.
        
        Args:
            direction (Union[str, List[float]]): Direction for k-point sampling (e.g., 'x').
            bz_coverage (float): Coverage of the Brillouin zone.
            n_k (int): Number of k-points.
            lattice_parameter (Optional[float]): Lattice parameter if provided.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Array of k-point magnitudes and corresponding vectors.
        """
        if isinstance(direction, str):
            direction = direction.lower()
            if direction == 'x':
                dir_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                computed_a = np.linalg.norm(self.a1)
            elif direction == 'y':
                dir_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                computed_a = np.linalg.norm(self.a2)
            elif direction == 'z':
                dir_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                computed_a = np.linalg.norm(self.a3)
            else:
                raise ValueError(f"Unknown direction '{direction}'")
        else:
            dir_vector = np.array(direction, dtype=np.float32)
            norm = np.linalg.norm(dir_vector)
            if norm < 1e-12:
                raise ValueError("Direction vector is near zero length")
            dir_vector /= norm
            computed_a = np.linalg.norm(self.a1)

        if lattice_parameter is None:
            lattice_parameter = computed_a
            logger.info(f"No lattice parameter provided; using computed value: {lattice_parameter:.3f} Å")

        # Determine maximum k-value based on Brillouin zone coverage
        k_max = bz_coverage * (4 * np.pi / lattice_parameter)
        k_points = np.linspace(0, k_max, n_k, dtype=np.float32)
        k_vectors = np.outer(k_points, dir_vector).astype(np.float32)

        logger.info(f"Sampling k-path along '{direction}' with {n_k} points up to {k_max:.3f} (2π/Å)")
        return k_points, k_vectors

    def calculate_sd(self, k_points: np.ndarray, k_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the Spectral Displacement (SD) for the provided k-points.
        
        Args:
            k_points (np.ndarray): Array of k-point magnitudes (2π/Å).
            k_vectors (np.ndarray): Array of k-point vectors (2π/Å).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: SD, per-atom SD, and frequency array.
        """
        n_k = len(k_points)
        n_t = self.traj.n_frames
        n_atoms = self.traj.n_atoms

        # Calculate displacements from average positions
        disp = self.traj.positions - np.mean(self.traj.positions, axis=0).astype(np.float32)

        # Initialize arrays for global and per-atom SD
        sd = np.zeros((n_t, n_k, 3), dtype=np.complex64)
        sd_per_atom = np.zeros((n_t, n_k, n_atoms, 3), dtype=np.complex64)
        
        # Calculate timestep and frequencies using FFT
        dt_s = (self.traj.timesteps[1] - self.traj.timesteps[0]) * 1e-12  # Assuming timesteps are in picoseconds
        freqs = np.fft.fftfreq(n_t, d=dt_s).astype(np.float32) * 1e-12  # Convert to THz

        for i, k_vec in enumerate(tqdm(k_vectors, total=n_k, desc="Computing SD", ncols=80)):
            # Calculate phase factors based on initial positions and k-vector
            phases = np.exp(-1j * np.dot(self.traj.positions[0], k_vec)).astype(np.complex64)
            
            # Apply phases to displacements for each atom
            uk_t = disp * phases[np.newaxis, :, np.newaxis]
            
            # Compute FFT for each atom and component
            uk_w = np.fft.fft(uk_t, axis=0)  # Perform FFT along the time axis
            
            # Store per-atom SD
            sd_per_atom[:, i, :, :] = uk_w.astype(np.complex64)
            
            # Sum over atoms for global SD, normalized by sqrt(N)
            sd[:, i, :] = uk_w.sum(axis=1) / np.sqrt(n_atoms)
        
        return sd, sd_per_atom, freqs

    def filter_sd(self, sd: np.ndarray, freqs: np.ndarray, k_points: np.ndarray,
                 freq_range: Optional[Tuple[float, float]] = None,
                 k_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Applies Gaussian filters to the SD based on frequency and k-point ranges.
        
        Args:
            sd (np.ndarray): The SD data to filter.
            freqs (np.ndarray): Frequency array.
            k_points (np.ndarray): k-point magnitudes (2π/Å).
            freq_range (Optional[Tuple[float, float]]): Frequency range (min, max) for filtering.
            k_range (Optional[Tuple[float, float]]): k-point range (min, max) for filtering.

        Returns:
            np.ndarray: Filtered SD data.
        """
        filtered = sd.copy()
        initial_nonzero = np.count_nonzero(filtered)
        
        # Apply Gaussian frequency filter if specified
        if freq_range is not None:
            f_min, f_max = freq_range
            f_center = (f_max + f_min) / 2
            f_sigma = (f_max - f_min) / 6  # Assuming 99.7% coverage within [f_min, f_max]
            
            freq_window = np.exp(-0.5 * ((freqs - f_center) / f_sigma)**2).astype(np.float32)
            # Reshape for broadcasting over SD dimensions
            freq_window = freq_window.reshape(-1, 1, *(1,) * (len(sd.shape) - 2))
            
            filtered *= freq_window
            logger.info(f"Applied Gaussian frequency filter: {f_center:.2f} THz ± {f_sigma:.2f} THz")
            
        # Apply Gaussian k-point filter if specified
        if k_range is not None:
            k_min, k_max = k_range
            k_center = (k_max + k_min) / 2
            k_sigma = (k_max - k_min) / 6  # Assuming 99.7% coverage within [k_min, k_max]
            
            k_window = np.exp(-0.5 * ((k_points - k_center) / k_sigma)**2).astype(np.float32)
            # Reshape for broadcasting over SD dimensions
            k_window = k_window.reshape(1, -1, *(1,) * (len(sd.shape) - 2))
            
            filtered *= k_window
            logger.info(f"Applied Gaussian k-point filter: {k_center:.3f} (2π/Å) ± {k_sigma:.3f} (2π/Å)")

        final_nonzero = np.count_nonzero(filtered)
        
        return filtered

    def plot_sd(self, sd: np.ndarray, freqs: np.ndarray, k_points: np.ndarray,
               output: str, cmap: str = 'inferno', vmin: float = -6.5,
               vmax: float = 0, global_max_intensity: Optional[float] = None,
               highlight_region: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """
        Generates and saves a plot of the Spectral Displacement (SD).
        
        Args:
            sd (np.ndarray): The SD data to plot.
            freqs (np.ndarray): Frequency array.
            k_points (np.ndarray): k-point magnitudes (2π/Å).
            output (str): Output filename for the plot.
            cmap (str): Colormap for the plot.
            vmin (float): Minimum value for color scaling.
            vmax (float): Maximum value for color scaling.
            global_max_intensity (Optional[float]): Global maximum intensity for normalization.
            highlight_region (Optional[Dict[str, Tuple[float, float]]]): Region to highlight on the plot.
        """
        try:
            # Determine intensity based on SD type (global or per-atom)
            if len(sd.shape) == 4:  # per-atom SD
                intensity = np.abs(np.sum(np.abs(sd)**2, axis=(2,3))).astype(np.float32)
            else:  # global SD
                intensity = np.abs(np.sum(sd * np.conj(sd), axis=-1)).astype(np.float32)
                
            # Normalize intensity
            if global_max_intensity is not None:
                max_intensity = global_max_intensity
            else:
                max_intensity = np.max(intensity)
            
            if max_intensity > 0:
                intensity /= max_intensity
            else:
                logger.warning("Maximum intensity is zero or negative; skipping normalization.")
            
            # Sort frequencies and corresponding intensity for plotting
            abs_freqs = np.abs(freqs).astype(np.float32)
            sorted_indices = np.argsort(abs_freqs)
            sorted_freqs = abs_freqs[sorted_indices]
            sorted_intensity = intensity[sorted_indices]
            
            # Apply logarithmic scaling to intensity for better visualization
            log_intensity = np.log10(sorted_intensity + 1e-10).astype(np.float32)
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            pcm = plt.pcolormesh(
                k_points, sorted_freqs, log_intensity,
                shading='gouraud',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            
            # Highlight a specific region if requested
            if highlight_region is not None:
                freq_range = highlight_region.get('freq_range')
                k_range = highlight_region.get('k_range')
                
                if freq_range is not None and k_range is not None:
                    from matplotlib.patches import Rectangle
                    f_min, f_max = freq_range
                    k_min, k_max = k_range
                    
                    # Draw a rectangle around the highlighted region
                    rect = Rectangle(
                        (k_min, f_min),
                        k_max - k_min,
                        f_max - f_min,
                        fill=False,
                        edgecolor='white',
                        linestyle='--',
                        linewidth=2
                    )
                    plt.gca().add_patch(rect)
                    
                    # Annotate the highlighted region
                    plt.text(
                        k_max + 0.05, (f_max + f_min)/2,
                        f'Selected\nRegion\n{f_min}-{f_max} THz\n{k_min}-{k_max} (2π/Å)',
                        color='white',
                        va='center',
                        fontsize=8
                    )
            
            # Labeling and aesthetics
            plt.xlabel('k (2π/Å)')
            plt.ylabel('Frequency (THz)')
            plt.ylim(0, np.max(sorted_freqs))
            plt.title('Spectral Displacement')
            plt.colorbar(pcm, label='log₁₀(Intensity)')
            
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SD plot saved as {output}")
        except Exception as e:
            logger.error(f"Failed to create SD plot: {e}")
            raise

class TrajectoryReconstructor:
    """
    Reconstructs atomic displacements from filtered SD data and exports them as a LAMMPS trajectory.
    """
    def __init__(self, traj: Trajectory, calculator: SDCalculator):
        """
        Initializes the reconstructor with trajectory data and an SD calculator.
        
        Args:
            traj (Trajectory): The original trajectory data.
            calculator (SDCalculator): The SD calculator instance.
        """
        self.traj = traj
        self.calculator = calculator
    
    def reconstruct_mode(
        self,
        sd_per_atom: np.ndarray,
        freqs: np.ndarray,
        k_points: np.ndarray,
        k_vectors: np.ndarray,
        desired_amplitude: Optional[float] = None
        ) -> np.ndarray:
        """
        Reconstructs atomic displacements from the filtered per-atom SD.
        
        Args:
            sd_per_atom (np.ndarray): Filtered per-atom SD data.
            freqs (np.ndarray): Frequency array.
            k_points (np.ndarray): k-point magnitudes (2π/Å).
            k_vectors (np.ndarray): k-point vectors (2π/Å).
            desired_amplitude (Optional[float]): Desired scaling factor for displacements.

        Returns:
            np.ndarray: Reconstructed displacements (frames, atoms, 3).
        """
        n_t = self.traj.n_frames
        n_atoms = self.traj.n_atoms
        reconstructed = np.zeros((n_t, n_atoms, 3), dtype=np.complex64)

        # Reference positions (equilibrium positions)
        ref_positions = np.mean(self.traj.positions, axis=0).astype(np.float32)
        
        # Pre-compute k·r phase terms for consistency
        k_dot_r = np.einsum('ka,na->kn', k_vectors, ref_positions)
        spatial_phases = np.exp(1j * k_dot_r).astype(np.complex64)
        
        logger.info("Reconstructing atomic displacements from SD data...")
        for ik, k_vec in enumerate(tqdm(k_vectors, desc="Reconstructing Trajectory", ncols=80)):
            # Initialize mode amplitudes for this k-point
            uk_mode = np.zeros((n_t, n_atoms, 3), dtype=np.complex64)
            
            # Process each spatial component (x, y, z)
            for comp in range(3):
                # Retrieve frequency components for this k-point and component
                uk_w = sd_per_atom[:, ik, :, comp].astype(np.complex64)
                
                # Perform inverse FFT to transform back to time domain
                uk_t = np.fft.ifft(uk_w, n=n_t, axis=0).astype(np.complex64)
                uk_mode[:, :, comp] = uk_t
            
            # Apply spatial phase to ensure correct displacement directions
            phase = spatial_phases[ik].reshape(1, -1, 1)
            reconstructed += uk_mode * phase
        
        # Average over all k-points
        reconstructed /= len(k_points)
        
        # Convert to real displacements
        reconstructed = np.real(reconstructed).astype(np.float32)
        
        # Optional scaling to match desired amplitude
        if desired_amplitude is not None:
            current_amplitude = np.sqrt(np.mean(np.sum(reconstructed**2, axis=2)))
            if current_amplitude > 1e-10:
                scale_factor = desired_amplitude / current_amplitude
                reconstructed *= scale_factor
                logger.info(f"Displacements scaled by a factor of {scale_factor:.3f}")
            else:
                logger.warning("Mean displacement too small for scaling; skipping amplitude adjustment.")
        
        return reconstructed.astype(np.float32)

    def write_lammps_trajectory(
        self,
        filename: str,
        displacements: np.ndarray,
        start_time: float = 0.0,
        timestep: Optional[float] = None
    ) -> None:
        """
        Writes the reconstructed displacements to a LAMMPS trajectory file.
        
        Args:
            filename (str): Output filename for the LAMMPS trajectory.
            displacements (np.ndarray): Reconstructed displacements (frames, atoms, 3).
            start_time (float): Starting timestep value.
            timestep (Optional[float]): Time interval between frames; if None, inferred from trajectory.
        """
        if timestep is None:
            timestep = self.traj.timesteps[1] - self.traj.timesteps[0]

        n_frames, n_atoms, _ = displacements.shape
        ref_positions = np.mean(self.traj.positions, axis=0).astype(np.float32)  # Reference positions
        box = self.traj.box

        # Extract tilt factors for triclinic boxes
        xy, xz, yz = box.tilts
        xlo_bound = 0.0
        xhi_bound = box.lengths[0]
        ylo_bound = 0.0
        yhi_bound = box.lengths[1]
        zlo_bound = 0.0
        zhi_bound = box.lengths[2]

        # Adjust box bounds based on tilts
        if xy != 0.0:
            xlo_bound = min(xlo_bound, xlo_bound + xy)
            xhi_bound = max(xhi_bound, xhi_bound + xy)
        if xz != 0.0:
            xlo_bound = min(xlo_bound, xlo_bound + xz)
            xhi_bound = max(xhi_bound, xhi_bound + xz)
        if yz != 0.0:
            ylo_bound = min(ylo_bound, ylo_bound + yz)
            yhi_bound = max(yhi_bound, yhi_bound + yz)

        logger.info(f"Writing reconstructed trajectory to {filename}")
        try:
            with open(filename, 'w') as f:
                for t in range(n_frames):
                    f.write("ITEM: TIMESTEP\n")
                    f.write(f"{int(start_time + t * timestep)}\n")
                    f.write("ITEM: NUMBER OF ATOMS\n")
                    f.write(f"{n_atoms}\n")

                    # Write box bounds with tilt factors
                    f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                    f.write(f"{xlo_bound:.6f} {xhi_bound:.6f} {xy:.6f}\n")
                    f.write(f"{ylo_bound:.6f} {yhi_bound:.6f} {xz:.6f}\n")
                    f.write(f"{zlo_bound:.6f} {zhi_bound:.6f} {yz:.6f}\n")

                    # Calculate absolute positions by adding displacements to reference positions
                    positions = ref_positions + displacements[t]

                    # Write atom data
                    f.write("ITEM: ATOMS id type x y z\n")
                    for i in range(n_atoms):
                        x, y, z = positions[i]
                        atom_type = self.traj.types[i]
                        f.write(f"{i+1} {atom_type} {x:.6f} {y:.6f} {z:.6f}\n")
            logger.info("Reconstructed trajectory successfully written.")
        except Exception as e:
            logger.error(f"Failed to write LAMMPS trajectory: {e}")
            raise

def main():
    """
    Main function to execute the SD analysis workflow.
    """
    parser = argparse.ArgumentParser(description='Spectral Displacement (SD) Analysis Tool')
    parser.add_argument('trajectory', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file', default=None)
    parser.add_argument('--reload', action='store_true', help='Force reloading the trajectory data from the original file')
    args = parser.parse_args()

    # Load configuration parameters from YAML file if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config file '{args.config}': {e}")
    
    # Extract configuration parameters with default values
    dt = config.get('dt', 0.005)  # Timestep in picoseconds
    nx = config.get('nx', 60)      # Grid points in x
    ny = config.get('ny', 60)      # Grid points in y
    nz = config.get('nz', 1)       # Grid points in z
    direction = config.get('direction', 'x')  # Direction for k-path
    bz_coverage = config.get('bz_coverage', 1.0)  # Brillouin zone coverage
    n_kpoints = config.get('n_kpoints', 60)        # Number of k-points
    wmin = config.get('wmin', 0)                   # Minimum frequency for filtering
    wmax = config.get('wmax', 50)                  # Maximum frequency for filtering
    kmin = config.get('kmin', None)                # Minimum k-point for filtering (2π/Å)
    kmax = config.get('kmax', None)                # Maximum k-point for filtering (2π/Å)
    amplitude = config.get('amplitude', 0.5)        # Desired amplitude scaling
    lattice_parameter = config.get('lattice_parameter', None)  # Optional lattice parameter

    logger.info("Initializing SD analysis...")
    loader = TrajectoryLoader(args.trajectory, dt=dt)
    traj = loader.load(reload=args.reload)

    # Display basic trajectory information
    logger.info(f"Trajectory loaded: {traj.n_frames} frames, {traj.n_atoms} atoms")
    logger.info(f"Simulation box lengths (Å): {traj.box.lengths}")
    logger.info(f"Simulation box tilts: {traj.box.tilts}")

    # Initialize SD calculator and reconstructor
    calculator = SDCalculator(traj, nx, ny, nz)
    reconstructor = TrajectoryReconstructor(traj, calculator)

    # Generate k-point path
    k_points, k_vectors = calculator.get_k_path(
        direction=direction,
        bz_coverage=bz_coverage,
        n_k=n_kpoints,
        lattice_parameter=lattice_parameter
    )

    # Calculate SD with per-atom information
    sd, sd_per_atom, freqs = calculator.calculate_sd(k_points, k_vectors)
    full_intensity = np.abs(np.sum(sd * np.conj(sd), axis=-1)).astype(np.float32)
    max_intensity = np.max(full_intensity)

    # Plot global SD
    calculator.plot_sd(
        sd, freqs, k_points,
        output='sd_global.png',
        vmin=-6.5,
        vmax=0,
        global_max_intensity=max_intensity
    )

    # Define frequency and k-point ranges for filtering
    freq_range = (wmin, wmax)
    # Ensure kmin and kmax are in 2π/Å units
    if kmin is not None and kmax is not None:
        k_range = (kmin, kmax)
    else:
        k_range = None

    # Apply filters to both global and per-atom SD
    filtered_sd = calculator.filter_sd(sd, freqs, k_points, freq_range=freq_range, k_range=k_range)
    filtered_sd_per_atom = calculator.filter_sd(sd_per_atom, freqs, k_points, freq_range=freq_range, k_range=k_range)

    # Plot filtered global SD with highlighted region
    calculator.plot_sd(
        filtered_sd, freqs, k_points,
        output='sd_filtered.png',
        vmin=-6.5,
        vmax=0,
        global_max_intensity=max_intensity,
        highlight_region={
            'freq_range': freq_range,
            'k_range': k_range
        }
    )

    # Reconstruct atomic displacements from filtered SD
    displacements = reconstructor.reconstruct_mode(
        filtered_sd_per_atom, freqs, k_points, k_vectors,
        desired_amplitude=amplitude
    )

    # Write the reconstructed trajectory to a LAMMPS trajectory file
    reconstructor.write_lammps_trajectory('reconstructed.lammpstrj', displacements)
    logger.info("SD analysis completed successfully.")

if __name__ == "__main__":
    main()
