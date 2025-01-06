"""
Spectral Displacement (SD) Analysis Tool

Memory-optimized version that uses memory mapping and chunked processing for large trajectories.
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
from typing import Tuple, List, Optional, Dict, Union, Generator, ContextManager
import logging
import argparse
import yaml
import os
import tempfile
import shutil
from contextlib import contextmanager
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

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
        self.filepath = Path(filename)
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
        self.traj = traj
        self.system_size = np.array([nx, ny, nz], dtype=np.int32)
        self.time_chunk_size = 1000  # Number of frames to process at once
        
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

        k_max = bz_coverage * (4 * np.pi / lattice_parameter)
        k_points = np.linspace(0, k_max, n_k, dtype=np.float32)
        k_vectors = np.outer(k_points, dir_vector).astype(np.float32)

        return k_points, k_vectors

    def _process_time_chunk(self, positions: np.ndarray, k_vectors: np.ndarray, 
                          mean_positions: np.ndarray) -> np.ndarray:
        """Process a chunk of trajectory frames following:
        f(t) = u(n,t) * exp( i k x̄(n) )
        Φ(k,ω) = | FFT{ Σn f(t) } |²
        where u(n,t) are the displacements from equilibrium positions
        """
        # Calculate displacements from mean positions
        displacements = positions - mean_positions[None, :, :]  # (time, atoms, 3)
        
        # Initialize output array for spectral components
        n_times = positions.shape[0]
        n_k = k_vectors.shape[0]
        sed = np.zeros((n_times, n_k, 3), dtype=np.complex64)
        
        # For each k-vector
        for ik, k in enumerate(k_vectors):
            # Phase factor from equilibrium positions
            phase = np.exp(1j * np.dot(mean_positions, k))  # (atoms,)
            
            # For each spatial component
            for alpha in range(3):
                # Displacement component times phase factor
                f_t = displacements[..., alpha] * phase[None, :]  # (time, atoms)
                # Sum over atoms
                sed[:, ik, alpha] = np.sum(f_t, axis=1)
                
        return sed
        
    def calculate_sd(self, k_points: np.ndarray, k_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Spectral Energy Density (SED) following:
        Φ(k,ω) = | ∫ Σn u°(n,t) exp( i k x̄(n) - i ω t ) dt |²
        """
        n_t = self.traj.n_frames
        n_k = len(k_points)
        
        # Calculate equilibrium positions
        mean_positions = np.mean(self.traj.positions, axis=0)
        
        # Initialize array for spectral components
        sed = np.zeros((n_t, n_k, 3), dtype=np.complex64)
        
        # Process trajectory in time chunks with progress bar
        chunks = list(range(0, n_t, self.time_chunk_size))
        pbar = tqdm(chunks, desc="Processing frames", unit="chunk")
        for chunk_start in pbar:
            chunk_end = min(chunk_start + self.time_chunk_size, n_t)
            positions_chunk = self.traj.positions[chunk_start:chunk_end]
            
            # Process this time chunk
            sed_chunk = self._process_time_chunk(positions_chunk, k_vectors, mean_positions)
            sed[chunk_start:chunk_end] = sed_chunk
            
            del positions_chunk, sed_chunk
        
        # Perform temporal FFT
        sed_w = np.fft.fft(sed, axis=0)
        
        # Calculate power spectrum (sum over spatial components)
        power_spectrum = np.sum(np.real(sed_w * np.conj(sed_w)), axis=-1)
        
        # Calculate frequencies
        dt_s = (self.traj.timesteps[1] - self.traj.timesteps[0]) * 1e-12
        freqs = np.fft.fftfreq(n_t, d=dt_s) * 1e-12
        
        return power_spectrum, freqs

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
            
            freq_window = np.exp(-0.5 * ((freqs - f_center) / f_sigma)**2).astype(np.float32)
            freq_window = freq_window.reshape(-1, 1, *(1,) * (len(sd.shape) - 2))
            
            filtered *= freq_window
            logger.info(f"Applied frequency filter: {f_center:.2f} THz ± {f_sigma:.2f} THz")
        
        # Apply Gaussian k-point filter if specified
        if k_range is not None:
            k_min, k_max = k_range
            k_center = (k_max + k_min) / 2
            k_sigma = (k_max - k_min) / 6
            
            k_window = np.exp(-0.5 * ((k_points - k_center) / k_sigma)**2).astype(np.float32)
            k_window = k_window.reshape(1, -1, *(1,) * (len(sd.shape) - 2))
            
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
            # Ensure sed is 2D (frequencies × k-points)
            if len(sed.shape) > 2:
                intensity = np.sum(sed, axis=tuple(range(2, len(sed.shape)))).astype(np.float32)
            else:
                intensity = sed.astype(np.float32)
            
            # Create meshgrid for plotting
            k_mesh, freq_mesh = np.meshgrid(k_points, np.abs(freqs))
            
            # Normalize and process intensity data
            if global_max_intensity is not None:
                max_intensity = global_max_intensity
            else:
                max_intensity = np.max(intensity)
            
            if max_intensity > 0:
                intensity /= max_intensity
            
            # Apply square root scaling for less extreme nonlinearity
            sqrt_intensity = np.sqrt(intensity + 1e-20).astype(np.float32)
            
            # Auto-determine vmin/vmax if not provided
            if vmin is None:
                vmin = np.percentile(sqrt_intensity[sqrt_intensity > 0], 1)
            if vmax is None:
                vmax = np.percentile(sqrt_intensity[sqrt_intensity > 0], 99)
            
            # Create the plot
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
                    from matplotlib.patches import Rectangle
                    f_min, f_max = freq_range
                    k_min, k_max = k_range
                    
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
            plt.title('Spectral Displacement')
            plt.colorbar(pcm, label='√Intensity')
            
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SD plot saved as {output}")
            
        except Exception as e:
            logger.error(f"Failed to create SD plot: {e}")
            raise
@contextmanager
def get_temp_dir() -> ContextManager[str]:
    """Creates and manages a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

class TrajectoryReconstructor:
    """Reconstructs atomic displacements from filtered SD data."""
    def __init__(self, traj: Trajectory, calculator: SDCalculator):
        self.traj = traj
        self.calculator = calculator
        self.time_chunk_size = 1000
        self.chunk_size = 1000

    def reconstruct_mode(self, sd: np.ndarray, freqs: np.ndarray,
                        k_points: np.ndarray, k_vectors: np.ndarray,
                        desired_amplitude: Optional[float] = None) -> np.ndarray:
        """Reconstruct atomic displacements with improved memory efficiency."""
        with get_temp_dir() as temp_dir:
            self.temp_dir = temp_dir
            n_t = self.traj.n_frames
            n_atoms = self.traj.n_atoms
            
            # Create memory-mapped array in temp directory
            temp_reconstructed_file = os.path.join(temp_dir, 'temp_reconstructed.mmap')
            reconstructed = np.memmap(temp_reconstructed_file, dtype='float32',
                                    mode='w+', shape=(n_t, n_atoms, 3))
            
            # Reference positions
            ref_positions = np.mean(self.traj.positions, axis=0)
            
            # Process k-points in chunks
            for k_chunk_start in range(0, len(k_points), self.chunk_size):
                k_chunk_end = min(k_chunk_start + self.chunk_size, len(k_points))
                k_chunk = slice(k_chunk_start, k_chunk_end)
                
                # Calculate phases for this chunk
                k_dot_r = np.einsum('ka,na->kn', 
                                  k_vectors[k_chunk], 
                                  ref_positions)
                spatial_phases = np.exp(1j * k_dot_r).astype(np.complex64)
                
                # Process time in chunks
                for t_chunk_start in range(0, n_t, self.chunk_size):
                    t_chunk_end = min(t_chunk_start + self.chunk_size, n_t)
                    t_chunk = slice(t_chunk_start, t_chunk_end)
                    
                    chunk_reconstructed = np.zeros((t_chunk_end - t_chunk_start,
                                                 n_atoms, 3), dtype=np.complex64)
                    
                    # Process each k-point in this chunk
                    for ik, k_vec in enumerate(k_vectors[k_chunk]):
                        # Get the frequency components for this k-point
                        uk_w = np.sqrt(sd[t_chunk, k_chunk_start + ik])  # Take sqrt of power spectrum
                        
                        # Add random phases for time evolution
                        phases = np.exp(2j * np.pi * np.random.random(uk_w.shape))
                        uk_w = uk_w * phases
                        
                        # Ensure Hermitian symmetry for real-valued output
                        n_freq = len(uk_w)
                        if n_freq % 2 == 0:  # Even number of frequencies
                            uk_w[n_freq//2+1:] = np.conj(uk_w[1:n_freq//2][::-1])
                        else:  # Odd number of frequencies
                            uk_w[(n_freq+1)//2:] = np.conj(uk_w[1:(n_freq-1)//2+1][::-1])
                        
                        # Transform back to time domain
                        uk_t = np.fft.ifft(uk_w, axis=0)
                        
                        # Create displacement field for all atoms
                        uk_mode = np.zeros((t_chunk_end - t_chunk_start,
                                          n_atoms, 3), dtype=np.complex64)
                        
                        # Project displacement along k-vector direction
                        k_norm = np.linalg.norm(k_vec)
                        if k_norm > 1e-10:
                            k_dir = k_vec / k_norm
                            for comp in range(3):
                                uk_mode[:, :, comp] = uk_t[:, np.newaxis] * k_dir[comp]
                        
                        # Apply spatial phase
                        phase = spatial_phases[ik].reshape(1, -1, 1)
                        chunk_reconstructed += uk_mode * phase
                        
                        del uk_mode
                    
                    # Convert to real displacements
                    reconstructed[t_chunk] = np.real(chunk_reconstructed)
                    
                    del chunk_reconstructed
            
            # Scale amplitude if requested
            if desired_amplitude is not None:
                current_amplitude = np.sqrt(np.mean(np.sum(reconstructed**2, axis=2)))
                if current_amplitude > 1e-10:
                    reconstructed *= (desired_amplitude / current_amplitude)
            
            return reconstructed

    def write_lammps_trajectory(self, filename: str,
                              displacements: np.ndarray,
                              start_time: float = 0.0,
                              timestep: Optional[float] = None) -> None:
        """Writes reconstructed displacements to a LAMMPS trajectory file."""
        if timestep is None:
            timestep = self.traj.timesteps[1] - self.traj.timesteps[0]

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

        # Write trajectory in chunks to save memory
        logger.info(f"Writing reconstructed trajectory to {filename}")
        try:
            with open(filename, 'w') as f:
                for chunk_start in range(0, n_frames, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, n_frames)
                    
                    for t in range(chunk_start, chunk_end):
                        f.write("ITEM: TIMESTEP\n")
                        f.write(f"{int(start_time + t * timestep)}\n")
                        f.write("ITEM: NUMBER OF ATOMS\n")
                        f.write(f"{n_atoms}\n")
                        f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                        f.write(f"{xlo_bound:.6f} {xhi_bound:.6f} {xy:.6f}\n")
                        f.write(f"{ylo_bound:.6f} {yhi_bound:.6f} {xz:.6f}\n")
                        f.write(f"{zlo_bound:.6f} {zhi_bound:.6f} {yz:.6f}\n")
                        
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
    parser = argparse.ArgumentParser(description='Memory-Optimized Spectral Displacement (SD) Analysis Tool')
    parser.add_argument('trajectory', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--reload', action='store_true', help='Force reloading the trajectory')
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
    amplitude = config.get('amplitude', 5.0)  # Increased default amplitude to make motion more visible
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

        # Save output plots to the specified directory
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

        # Plot filtered SD with adjusted intensity scaling for better visibility
        filtered_intensity = np.sum(filtered_sd, axis=-1)
        filtered_max = np.max(filtered_intensity)
        if filtered_max > 0:
            filtered_sd = filtered_sd * (max_intensity / filtered_max)  # Rescale to match global intensity

        calculator.plot_sed(
            filtered_sd, freqs, k_points,
            output=str(filtered_plot),
            global_max_intensity=max_intensity,
            max_freq=max_freq,
            vmin=np.sqrt(1e-6),   # Adjusted for better visibility
            vmax=np.sqrt(1.0),    # Maximum normalized value
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
