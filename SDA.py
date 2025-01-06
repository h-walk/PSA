"""
Spectral Displacement (SD) Analysis Tool

Computes the SD, applies filtering, reconstructs atomic displacements, and outputs
results as plots and a LAMMPS-compatible trajectory file.

Dependencies:
- numpy
- ovito
- matplotlib
- tqdm
- yaml
"""

import numpy as np
from pathlib import Path
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Union
import logging
import argparse
import yaml
import tempfile
import shutil
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for plotting
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.error(f"Failed to import matplotlib: {e}")
    logger.error("Please ensure matplotlib is installed: pip install matplotlib")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

from ovito.io import import_file

@dataclass
class Box:
    """Represents the simulation box with lengths, tilts, and the full cell matrix."""
    lengths: np.ndarray  # [lx, ly, lz]
    tilts: np.ndarray    # [xy, xz, yz]
    matrix: np.ndarray   # full 3x3 cell matrix

    @classmethod
    def from_ovito(cls, cell) -> 'Box':
        """Constructs a Box instance from an OVITO cell object."""
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
        """Converts the Box instance to a dictionary for saving."""
        return {
            'lengths': self.lengths,
            'tilts': self.tilts,
            'matrix': self.matrix
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Box':
        """Constructs a Box instance from a dictionary."""
        return cls(
            lengths=np.array(data['lengths'], dtype=np.float32),
            tilts=np.array(data['tilts'], dtype=np.float32),
            matrix=np.array(data['matrix'], dtype=np.float32)
        )

@dataclass
class Trajectory:
    """Stores trajectory data including positions, atom types, timesteps, and simulation box."""
    positions: np.ndarray  # Shape: (frames, atoms, 3)
    types: np.ndarray      # Shape: (atoms,)
    timesteps: np.ndarray  # Shape: (frames,)
    box: Box
    
    def __post_init__(self):
        """Validates the dimensions of the trajectory data."""
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
    """Handles loading of trajectory data."""
    def __init__(self, filename: str, dt: float = 1.0):
        if dt <= 0:
            raise ValueError("Time step (dt) must be positive")
        self.filepath = Path(filename)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filename}")
        self.dt = dt
    
    def load(self) -> Trajectory:
        """Loads trajectory data."""
        try:
            logger.info("Loading trajectory...")
            pipeline = import_file(str(self.filepath))
            n_frames = pipeline.source.num_frames
            
            # Load first frame to get atom count and box
            frame0 = pipeline.compute(0)
            n_atoms = len(frame0.particles.positions)
            
            # Initialize arrays
            positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
            
            # Load all frames with progress bar
            for i in tqdm(range(n_frames), desc="Loading frames", unit="frame"):
                frame = pipeline.compute(i)
                positions[i] = frame.particles.positions.array.astype(np.float32)
            
            types = frame0.particles.particle_types.array
            timesteps = np.arange(n_frames, dtype=np.float32) * self.dt
            box = Box.from_ovito(frame0.cell)
            
            return Trajectory(positions, types, timesteps, box)
                
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            raise

class SDCalculator:
    """Calculates the Spectral Displacement (SD) from trajectory data."""
    def __init__(self, traj: Trajectory, nx: int, ny: int, nz: int):
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("System size dimensions must be positive integers")
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

    def get_k_path(self, direction: Union[str, List[float]], bz_coverage: float, n_k: int, 
                   lattice_parameter: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a path of k-points along a specified direction."""
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
            dir_vector /= np.linalg.norm(dir_vector)
            computed_a = np.linalg.norm(self.a1)

        if lattice_parameter is None:
            lattice_parameter = computed_a
            logger.info(f"Using computed lattice parameter: {lattice_parameter:.3f} Å")

        k_max = bz_coverage * (2 * np.pi / lattice_parameter)
        k_points = np.linspace(0, k_max, n_k, dtype=np.float32)
        k_vectors = np.outer(k_points, dir_vector).astype(np.float32)

        return k_points, k_vectors

    def calculate_sd(self, k_points: np.ndarray, k_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the Spectral Energy Density (SED)."""
        n_t = self.traj.n_frames
        n_k = len(k_points)
        
        # Calculate mean positions
        mean_positions = np.mean(self.traj.positions, axis=0)
        
        # Initialize arrays
        sed = np.zeros((n_t, n_k, 3), dtype=np.complex64)
        
        # Process trajectory with progress bar
        for ik, k in enumerate(tqdm(k_vectors, desc="Processing k-points", unit="k-point")):
            # Phase factor from equilibrium positions
            phase = np.exp(1j * np.dot(mean_positions, k))  # (atoms,)
            
            # Calculate displacements and phases for all times at once
            displacements = self.traj.positions - mean_positions[None, :, :]  # (time, atoms, 3)
            
            # For each spatial component
            for alpha in range(3):
                # Displacement component times phase factor for all times
                f_t = displacements[..., alpha] * phase[None, :]  # (time, atoms)
                # Sum over atoms
                sed[:, ik, alpha] = np.sum(f_t, axis=1)
        
        # Perform temporal FFT
        sed_w = np.fft.fft(sed, axis=0)
        
        # Calculate frequencies
        dt_s = (self.traj.timesteps[1] - self.traj.timesteps[0]) * 1e-12  # Convert ps to s
        freqs = np.fft.fftfreq(n_t, d=dt_s) * 1e-12  # Convert to THz
        
        return sed_w, freqs

    def filter_sd(self, sd: np.ndarray, freqs: np.ndarray, k_points: np.ndarray,
                 freq_range: Optional[Tuple[float, float]] = None,
                 k_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Applies Gaussian filters to the SD based on frequency and k-point ranges."""
        filtered = sd.copy()
        
        # Apply Gaussian frequency filter if specified
        if freq_range is not None:
            f_min, f_max = freq_range
            f_center = (f_max + f_min) / 2
            f_sigma = (f_max - f_min) / 6
            
            freq_window = np.exp(-0.5 * ((freqs - f_center) / f_sigma)**2)
            freq_window = freq_window.reshape(-1, 1, 1)
            
            filtered *= freq_window
            logger.info(f"Applied frequency filter: {f_center:.2f} THz ± {f_sigma:.2f} THz")
        
        # Apply Gaussian k-point filter if specified
        if k_range is not None:
            k_min, k_max = k_range
            k_center = (k_max + k_min) / 2
            k_sigma = (k_max - k_min) / 6
            
            k_window = np.exp(-0.5 * ((k_points - k_center) / k_sigma)**2)
            k_window = k_window.reshape(1, -1, 1)
            
            filtered *= k_window
            logger.info(f"Applied k-point filter: {k_center:.3f} (2π/Å) ± {k_sigma:.3f} (2π/Å)")
        
        return filtered

    def plot_sed(self, sed: np.ndarray, freqs: np.ndarray, k_points: np.ndarray,
                output: str, cmap: str = 'inferno', vmin: Optional[float] = None,
                vmax: Optional[float] = None, global_max_intensity: Optional[float] = None,
                highlight_region: Optional[Dict[str, Tuple[float, float]]] = None,
                max_freq: Optional[float] = None) -> None:
        """Generates and saves a plot of the SED."""
        try:
            # Take positive frequencies only
            pos_freq_mask = freqs >= 0
            freqs = freqs[pos_freq_mask]
            sed = sed[pos_freq_mask]

            # Process data for plotting
            if len(sed.shape) > 2:
                # Sum over components (alpha)
                intensity = np.abs(sed).sum(axis=-1)
            else:
                intensity = np.abs(sed)
            
            # Create meshgrid for plotting
            k_mesh, freq_mesh = np.meshgrid(k_points, freqs)
            
            # Ensure we have real values
            intensity = np.real(intensity)
            
            # Normalize intensity
            if global_max_intensity is not None:
                max_intensity = global_max_intensity
            else:
                max_intensity = np.max(intensity)
            
            if max_intensity > 0:
                intensity = intensity / max_intensity
            
            # Apply square root scaling for better visualization
            sqrt_intensity = np.sqrt(np.abs(intensity + 1e-20))
            
            # Auto-determine vmin/vmax if not provided
            if vmin is None:
                vmin = np.percentile(sqrt_intensity[sqrt_intensity > 0], 1)
            if vmax is None:
                vmax = np.percentile(sqrt_intensity[sqrt_intensity > 0], 99)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            pcm = plt.pcolormesh(
                k_mesh, freq_mesh, sqrt_intensity,
                shading='gouraud',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            
            # Add highlighted region if specified
            if highlight_region is not None:
                freq_range = highlight_region.get('freq_range')
                k_range = highlight_region.get('k_range')
                
                if freq_range is not None and k_range is not None:
                    f_min, f_max = freq_range
                    k_min, k_max = k_range
                    
                    rect = plt.Rectangle(
                        (k_min, f_min),
                        k_max - k_min,
                        f_max - f_min,
                        fill=False,
                        edgecolor='white',
                        linestyle='--',
                        linewidth=2
                    )
                    plt.gca().add_patch(rect)
                    
                    plt.text(
                        k_max + 0.05, (f_max + f_min)/2,
                        f'Selected\nRegion\n{f_min}-{f_max} THz\n{k_min}-{k_max} (2π/Å)',
                        color='white',
                        va='center',
                        fontsize=8
                    )
            
            plt.xlabel('k (2π/Å)')
            plt.ylabel('Frequency (THz)')
            freq_max = max_freq if max_freq is not None else np.max(freqs)
            plt.ylim(0, freq_max)
            plt.title('Spectral Energy Density')
            plt.colorbar(pcm, label='√Intensity (arb. units)')
            
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SED plot saved as {output}")
            
        except Exception as e:
            logger.error(f"Failed to create SED plot: {e}")
            raise


class TrajectoryReconstructor:
    """Reconstructs atomic displacements from filtered SD data."""
    def __init__(self, traj: Trajectory, calculator: SDCalculator):
        self.traj = traj
        self.calculator = calculator

    def reconstruct_mode(self, sd: np.ndarray, freqs: np.ndarray,
                        k_points: np.ndarray, k_vectors: np.ndarray,
                        desired_amplitude: Optional[float] = 1.0) -> np.ndarray:
        """Reconstruct atomic displacements from filtered SD data.
        
        Args:
            sd: Spectral density data
            freqs: Frequency values in THz
            k_points: k-point magnitudes
            k_vectors: k-vectors in reciprocal space
            desired_amplitude: Desired RMS displacement amplitude in Angstroms (default: 1.0 Å)
            
        Returns:
            Array of atomic displacements with shape (n_frames, n_atoms, 3)
        """
        n_t = self.traj.n_frames
        n_atoms = self.traj.n_atoms
        
        # Initialize output array for displacements
        reconstructed = np.zeros((n_t, n_atoms, 3), dtype=np.float32)
        
        # Reference positions (mean positions)
        ref_positions = np.mean(self.traj.positions, axis=0)
        
        # Time array for phase evolution (in picoseconds)
        dt = self.traj.timesteps[1] - self.traj.timesteps[0]  # ps
        time = np.arange(n_t) * dt
        
        # Process each k-point and frequency
        for ik, k_vec in enumerate(k_vectors):
            k_norm = np.linalg.norm(k_vec)
            if k_norm < 1e-10:
                continue
                
            # Normalized k-vector for displacement direction
            k_dir = k_vec / k_norm
            
            # Calculate phases for all atoms at this k-point
            k_dot_r = np.dot(ref_positions, k_vec)
            spatial_phases = np.exp(1j * k_dot_r)
            
            # Get frequency components for this k-point
            for i_freq, (freq, amplitudes) in enumerate(zip(freqs, sd[:, ik])):
                # Skip if frequency or amplitude is too small
                if abs(freq) < 1e-10 or not np.any(np.abs(amplitudes) > 1e-10):
                    continue
                    
                # Convert frequency to radians/ps for time evolution
                omega = 2 * np.pi * freq  # rad/ps (since freq is in THz)
                
                # Time evolution
                time_phase = np.exp(1j * omega * time)
                
                # Add random initial phase
                init_phase = np.exp(2j * np.pi * np.random.random())
                
                # Combine all phase factors
                total_phase = init_phase * time_phase.reshape(-1, 1) * spatial_phases
                
                # Add contribution to displacement field for each component
                for alpha in range(3):
                    disp = np.real(total_phase * amplitudes[alpha] * k_dir[alpha])
                    reconstructed[:, :, alpha] += disp
        
        # Scale to desired amplitude if specified
        if desired_amplitude is not None:
            current_rms = np.sqrt(np.mean(np.sum(reconstructed**2, axis=2)))
            if current_rms > 1e-10:
                scale_factor = desired_amplitude / current_rms
                reconstructed *= scale_factor
        
        return reconstructed

    def write_lammps_trajectory(self, filename: str, displacements: np.ndarray,
                              start_time: int = 0, timestep: int = 1) -> None:
        """Write reconstructed trajectory in LAMMPS format.
        
        Args:
            filename: Output trajectory filename
            displacements: Array of atomic displacements
            start_time: Initial timestep number
            timestep: Timestep increment
        """
        n_frames, n_atoms, _ = displacements.shape
        ref_positions = np.mean(self.traj.positions, axis=0)
        box = self.traj.box

        # Calculate box bounds following LAMMPS triclinic box convention
        xy, xz, yz = box.tilts
        xlo = 0.0
        xhi = box.lengths[0]
        ylo = 0.0
        yhi = box.lengths[1]
        zlo = 0.0
        zhi = box.lengths[2]

        # Adjust bounds for triclinic box
        xlo_bound = xlo + min(0.0, xy, xz, xy+xz)
        xhi_bound = xhi + max(0.0, xy, xz, xy+xz)
        ylo_bound = ylo + min(0.0, yz)
        yhi_bound = yhi + max(0.0, yz)
        zlo_bound = zlo
        zhi_bound = zhi

        # Write trajectory
        logger.info(f"Writing reconstructed trajectory to {filename}")
        try:
            with open(filename, 'w') as f:
                for t in tqdm(range(n_frames), desc="Writing trajectory", unit="frame"):
                    f.write("ITEM: TIMESTEP\n")
                    f.write(f"{int(start_time + t * timestep)}\n")
                    f.write("ITEM: NUMBER OF ATOMS\n")
                    f.write(f"{n_atoms}\n")
                    f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                    f.write(f"{xlo_bound:.6f} {xhi_bound:.6f} {xy:.6f}\n")
                    f.write(f"{ylo_bound:.6f} {yhi_bound:.6f} {xz:.6f}\n")
                    f.write(f"{zlo_bound:.6f} {zhi_bound:.6f} {yz:.6f}\n")
                    
                    # Calculate positions for this frame
                    positions = ref_positions + displacements[t]
                    
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
    parser = argparse.ArgumentParser(description='Spectral Displacement (SD) Analysis Tool')
    parser.add_argument('trajectory', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, help='Output directory for results',
                       default='.')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    # Set parameters
    dt = config.get('dt', 0.005)
    nx = config.get('nx', 60)
    ny = config.get('ny', 60)
    nz = config.get('nz', 1)
    direction = config.get('direction', 'x')
    bz_coverage = config.get('bz_coverage', 1.0)
    n_kpoints = config.get('n_kpoints', 60)
    wmin = config.get('wmin', 0)
    wmax = config.get('wmax', 50)
    max_freq = config.get('max_freq', None)
    kmin = config.get('kmin', None)
    kmax = config.get('kmax', None)
    amplitude = config.get('amplitude', 1.0)  # Default amplitude of 1.0 Å
    lattice_parameter = config.get('lattice_parameter', None)

    try:
        # Initialize components
        loader = TrajectoryLoader(args.trajectory, dt=dt)
        traj = loader.load()
        calculator = SDCalculator(traj, nx, ny, nz)
        
        # Generate k-points and k-vectors
        k_points, k_vectors = calculator.get_k_path(
            direction=direction,
            bz_coverage=bz_coverage,
            n_k=n_kpoints,
            lattice_parameter=lattice_parameter
        )
        
        # Calculate SD
        logger.info("Calculating SD...")
        power_spectrum, freqs = calculator.calculate_sd(k_points, k_vectors)
        
        # Use power spectrum directly
        sd = power_spectrum

        # Calculate global intensity for plotting
        full_intensity = np.sum(sd, axis=-1)
        max_intensity = np.max(full_intensity)

        # Save output plots
        global_plot = output_dir / 'sd_global.png'
        filtered_plot = output_dir / 'sd_filtered.png'
        output_traj = output_dir / 'reconstructed.lammpstrj'

        # Plot unfiltered SD
        calculator.plot_sed(
            sd, freqs, k_points,
            output=str(global_plot),
            global_max_intensity=max_intensity,
            max_freq=max_freq
        )

        # Apply filters
        freq_range = (wmin, wmax)
        k_range = (kmin, kmax) if kmin is not None and kmax is not None else None

        # Apply filters to SD
        filtered_sd = calculator.filter_sd(sd, freqs, k_points, 
                                        freq_range=freq_range, k_range=k_range)

        # Plot filtered SD (no rescaling needed - keep original intensities)
        calculator.plot_sed(
            filtered_sd, freqs, k_points,
            output=str(filtered_plot),
            global_max_intensity=max_intensity,
            max_freq=max_freq,
            vmin=np.sqrt(1e-6),
            vmax=np.sqrt(1.0),
            highlight_region={
                'freq_range': freq_range,
                'k_range': k_range
            }
        )

        # Reconstruct atomic displacements
        logger.info("Reconstructing atomic displacements...")
        reconstructor = TrajectoryReconstructor(traj, calculator)
        displacements = reconstructor.reconstruct_mode(
            filtered_sd, freqs, k_points, k_vectors,
            desired_amplitude=amplitude
        )

        # Write reconstructed trajectory
        reconstructor.write_lammps_trajectory(str(output_traj), displacements)
        
        logger.info("SD analysis completed successfully.")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
