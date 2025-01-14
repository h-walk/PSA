"""
Spectral Displacement (SD) Analysis Tool
Computes the SD, applies filtering, reconstructs atomic displacements, and outputs
results as plots and a LAMMPS-compatible trajectory file.

**MODIFIED** to:
1) Unwrap coords for PBC when loading.
2) Cast complex arrays to real floats before normalizing in plot_sed to avoid casting errors.

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
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as e:
    logging.error(f"Failed to import matplotlib: {e}")
    logging.error("Please ensure matplotlib is installed: pip install matplotlib")
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
from ovito.modifiers import UnwrapTrajectoriesModifier

@dataclass
class Box:
    """Represents the simulation box with lengths, tilts, and the full cell matrix."""
    lengths: np.ndarray
    tilts: np.ndarray
    matrix: np.ndarray
    
    @classmethod
    def from_ovito(cls, cell) -> 'Box':
        matrix = cell.matrix.copy().astype(np.float32)
        lengths = np.array([matrix[0,0], matrix[1,1], matrix[2,2]], dtype=np.float32)
        tilts = np.array([matrix[0,1], matrix[0,2], matrix[1,2]], dtype=np.float32)
        return cls(lengths=lengths, tilts=tilts, matrix=matrix)
        
    def to_dict(self) -> Dict:
        return {
            'lengths': self.lengths.tolist(),
            'tilts': self.tilts.tolist(),
            'matrix': self.matrix.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Box':
        return cls(
            lengths=np.array(data['lengths'], dtype=np.float32),
            tilts=np.array(data['tilts'], dtype=np.float32),
            matrix=np.array(data['matrix'], dtype=np.float32)
        )

@dataclass
class Trajectory:
    """
    Stores unwrapped positions, velocities, types, timesteps, and the simulation box.
    """
    positions: np.ndarray
    velocities: np.ndarray
    types: np.ndarray
    timesteps: np.ndarray
    box: Box
    
    def __post_init__(self):
        if self.positions.ndim != 3:
            raise ValueError("Positions must be a 3D array (frames, atoms, xyz)")
        if self.velocities.ndim != 3:
            raise ValueError("Velocities must be a 3D array (frames, atoms, xyz)")
        if self.types.ndim != 1:
            raise ValueError("Types must be a 1D array")
        if self.timesteps.ndim != 1:
            raise ValueError("Timesteps must be a 1D array")
    
    @property
    def n_frames(self) -> int:
        return len(self.timesteps)
    
    @property
    def n_atoms(self) -> int:
        return len(self.types)

class TrajectoryLoader:
    """
    Loads the trajectory, unwraps under PBC, and can save to .npy for faster reload.
    """
    def __init__(self, filename: str, dt: float = 1.0):
        if dt <= 0:
            raise ValueError("Time step (dt) must be positive")
        self.filepath = Path(filename)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filename}")
        self.dt = dt
    
    def load(self) -> Trajectory:
        """Unwrap positions, load velocities if present."""
        try:
            logger.info("Loading trajectory...")
            base_path = self.filepath.parent / self.filepath.stem
            npy_files = {
                'positions': base_path.with_suffix('.positions.npy'),
                'velocities': base_path.with_suffix('.velocities.npy'),
                'types': base_path.with_suffix('.types.npy'),
                'mean_positions': base_path.with_suffix('.mean_positions.npy'),
                'displacements': base_path.with_suffix('.displacements.npy')
            }
            
            if all(f.exists() for f in npy_files.values()):
                logger.info("Found pre-computed numpy arrays, loading...")
                try:
                    positions = np.load(npy_files['positions'])
                    velocities = np.load(npy_files['velocities'])
                    types = np.load(npy_files['types'])
                    timesteps = np.arange(len(positions), dtype=np.float32) * self.dt
                    
                    pipeline = import_file(str(self.filepath))
                    pipeline.modifiers.append(UnwrapTrajectoriesModifier())
                    frame0 = pipeline.compute(0)
                    box = Box.from_ovito(frame0.cell)
                    
                    logger.info("Successfully loaded pre-computed data")
                    return Trajectory(positions, velocities, types, timesteps, box)
                except Exception as e:
                    logger.warning(f"Failed to load numpy files: {e}")
                    logger.info("Falling back to loading from trajectory file")
            
            pipeline = import_file(str(self.filepath))
            pipeline.modifiers.append(UnwrapTrajectoriesModifier())
            
            n_frames = pipeline.source.num_frames
            frame0 = pipeline.compute(0)
            n_atoms = len(frame0.particles.positions)
            
            positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
            velocities = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
            
            for i in tqdm(range(n_frames), desc="Loading frames", unit="frame"):
                frame = pipeline.compute(i)
                positions[i] = frame.particles.positions.array.astype(np.float32)
                if hasattr(frame.particles, 'velocities'):
                    velocities[i] = frame.particles.velocities.array.astype(np.float32)
                else:
                    logger.warning("No velocities found in trajectory")
            
            types = frame0.particles.particle_types.array
            timesteps = np.arange(n_frames, dtype=np.float32) * self.dt
            box = Box.from_ovito(frame0.cell)
            
            return Trajectory(positions, velocities, types, timesteps, box)
                
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            raise
            
    def save_numpy_arrays(self, traj: Trajectory,
                          sd: Optional[np.ndarray] = None,
                          freqs: Optional[np.ndarray] = None,
                          use_velocities: bool = False) -> None:
        """Saves unwrapped positions, velocities, and optional SD data."""
        try:
            base_path = self.filepath.parent / self.filepath.stem
            np.save(base_path.with_suffix('.positions.npy'), traj.positions)
            np.save(base_path.with_suffix('.velocities.npy'), traj.velocities)
            np.save(base_path.with_suffix('.types.npy'), traj.types)
            
            mean_positions = np.mean(traj.positions, axis=0)
            displacements = traj.positions - mean_positions[None, :, :]
            
            np.save(base_path.with_suffix('.mean_positions.npy'), mean_positions)
            np.save(base_path.with_suffix('.displacements.npy'), displacements)
            
            if sd is not None and freqs is not None:
                data_type = 'vel' if use_velocities else 'disp'
                np.save(base_path.with_suffix(f'.sd_{data_type}.npy'), sd)
                np.save(base_path.with_suffix(f'.freqs_{data_type}.npy'), freqs)
            
            logger.info(f"Saved numpy arrays with base path: {base_path}")
        except Exception as e:
            logger.error(f"Failed to save numpy arrays: {e}")
            raise

class SDCalculator:
    """Calculates the Spectral Displacement (SD) from unwrapped trajectory data."""
    def __init__(self, traj: Trajectory, nx: int, ny: int, nz: int, use_velocities: bool = False):
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("System size dimensions must be positive integers")
        self.traj = traj
        self.system_size = np.array([nx, ny, nz], dtype=np.int32)
        self.use_velocities = use_velocities
        
        cell_mat = self.traj.box.matrix.astype(np.float32)
        self.a1 = cell_mat[:,0] / float(nx)
        self.a2 = cell_mat[:,1] / float(ny)
        self.a3 = cell_mat[:,2] / float(nz)
        
        volume = np.dot(self.a1, np.cross(self.a2, self.a3))
        b1 = 2 * np.pi * np.cross(self.a2, self.a3) / volume
        b2 = 2 * np.pi * np.cross(self.a3, self.a1) / volume
        b3 = 2 * np.pi * np.cross(self.a1, self.a2) / volume
        self.recip_vectors = np.vstack([b1, b2, b3]).astype(np.float32)
        
    def get_k_path(self, direction: Union[str, List[float]], bz_coverage: float, 
                   n_k: int, lattice_parameter: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(direction, str):
            direction_str = direction.lower()
            if direction_str == 'x':
                dir_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                computed_a = np.linalg.norm(self.a1)
            elif direction_str == 'y':
                dir_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                computed_a = np.linalg.norm(self.a2)
            elif direction_str == 'z':
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
        n_t = self.traj.n_frames
        n_k = len(k_points)
        mean_positions = np.mean(self.traj.positions, axis=0)
        sed = np.zeros((n_t, n_k, 3), dtype=np.complex64)
        
        if self.use_velocities:
            logger.info("Using velocities for SD calculation")
            data_array = self.traj.velocities
        else:
            logger.info("Using displacements for SD calculation")
            data_array = self.traj.positions - mean_positions[None, :, :]
        
        for ik, kvec in enumerate(tqdm(k_vectors, desc="Processing k-points", unit="k-point")):
            phase = np.exp(1j * np.dot(mean_positions, kvec))
            for alpha in range(3):
                tmp_arr = data_array[..., alpha] * phase[None, :]
                sed[:, ik, alpha] = np.sum(tmp_arr, axis=1)
        
        sed_w = np.fft.fft(sed, axis=0)
        
        dt_s = (self.traj.timesteps[1] - self.traj.timesteps[0]) * 1e-12
        freqs = np.fft.fftfreq(n_t, d=dt_s) * 1e-12
        
        return sed_w, freqs
    
    def filter_sd(self, sd: np.ndarray, freqs: np.ndarray, k_points: np.ndarray,
                  freq_range: Optional[Tuple[float, float]] = None,
                  k_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        filtered = sd.copy()
        
        if freq_range is not None:
            f_min, f_max = freq_range
            f_center = 0.5*(f_max + f_min)
            f_sigma = (f_max - f_min) / 6
            freq_window = np.exp(-0.5 * ((freqs - f_center)/f_sigma)**2)
            freq_window = freq_window.reshape(-1,1,1)
            filtered *= freq_window
            logger.info(f"Applied frequency filter: {f_center:.2f} THz ± {f_sigma:.2f} THz")
        
        if k_range is not None:
            k_min, k_max = k_range
            k_center = 0.5*(k_max + k_min)
            k_sigma = (k_max - k_min) / 6
            k_window = np.exp(-0.5 * ((k_points - k_center)/k_sigma)**2)
            k_window = k_window.reshape(1,-1,1)
            filtered *= k_window
            logger.info(f"Applied k-point filter: {k_center:.3f} (2π/Å) ± {k_sigma:.3f} (2π/Å)")
        
        return filtered
    
    def plot_sed(self, sed: np.ndarray, freqs: np.ndarray, k_points: np.ndarray,
                 output: str, cmap: str = 'inferno',
                 vmin: Optional[float] = None, vmax: Optional[float] = None,
                 global_max_intensity: Optional[float] = None,
                 highlight_region: Optional[Dict[str, Tuple[float, float]]] = None,
                 max_freq: Optional[float] = None) -> None:
        try:
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            sed = sed[pos_mask]
            
            # Sum over 3 components => shape (n_freq, n_k)
            intensity = np.abs(sed).sum(axis=-1)
            
            # Force real float
            intensity = intensity.real.astype(np.float32)
            
            k_mesh, f_mesh = np.meshgrid(k_points, freqs)
            
            if global_max_intensity is not None:
                max_intensity = global_max_intensity
            else:
                max_intensity = np.max(intensity)
            
            if max_intensity > 0:
                intensity = intensity / max_intensity  # safe float division
                
            sqrt_intensity = np.sqrt(freqs[:,np.newaxis]*intensity + 1e-20)
            
            if vmin is None:
                vmin = np.percentile(sqrt_intensity[sqrt_intensity>0], 1)
            if vmax is None:
                vmax = np.percentile(sqrt_intensity[sqrt_intensity>0], 99)
            
            plt.figure(figsize=(10,8))
            pcm = plt.pcolormesh(
                k_mesh, f_mesh, sqrt_intensity,
                shading='gouraud', cmap=cmap,
                vmin=vmin, vmax=vmax
            )
            
            if highlight_region:
                fr = highlight_region.get('freq_range')
                kr = highlight_region.get('k_range')
                if fr and kr:
                    f_min, f_max = fr
                    k_min, k_max = kr
                    rect = plt.Rectangle(
                        (k_min, f_min), k_max - k_min, f_max - f_min,
                        fill=False, edgecolor='white', linestyle='--', linewidth=2
                    )
                    plt.gca().add_patch(rect)
                    plt.text(
                        k_max + 0.05, 0.5*(f_max+f_min),
                        f'Selected\nRegion\n{f_min}-{f_max} THz\n{k_min}-{k_max} (2π/Å)',
                        color='white', va='center', fontsize=8
                    )
            
            plt.xlabel('k (2π/Å)')
            plt.ylabel('Frequency (THz)')
            if max_freq is not None:
                plt.ylim(0, max_freq)
            else:
                plt.ylim(0, np.max(freqs))
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
        n_t = self.traj.n_frames
        n_atoms = self.traj.n_atoms
        
        reconstructed = np.zeros((n_t, n_atoms, 3), dtype=np.float32)
        ref_positions = np.mean(self.traj.positions, axis=0)
        
        dt = self.traj.timesteps[1] - self.traj.timesteps[0]
        time = np.arange(n_t)*dt
        
        for ik, kvec in enumerate(k_vectors):
            k_norm = np.linalg.norm(kvec)
            if k_norm < 1e-10:
                continue
            k_dir = kvec / k_norm
            
            k_dot_r = np.dot(ref_positions, kvec)
            spatial_phase = np.exp(1j*k_dot_r)
            
            for i_freq, (freq, amp) in enumerate(zip(freqs, sd[:,ik])):
                if abs(freq)<1e-10 or not np.any(np.abs(amp)>1e-10):
                    continue
                omega = 2*np.pi*freq  # rad/ps
                t_phase = np.exp(1j*omega*time)
                init_phase = np.exp(2j*np.pi*np.random.random())
                total_phase = init_phase * t_phase.reshape(-1,1)*spatial_phase
                
                for alpha in range(3):
                    disp_ = np.real(total_phase * amp[alpha]*k_dir[alpha])
                    reconstructed[:,:,alpha]+= disp_
        
        if desired_amplitude is not None:
            current_rms = np.sqrt(np.mean(np.sum(reconstructed**2, axis=2)))
            if current_rms>1e-10:
                scale_factor = desired_amplitude/current_rms
                reconstructed *= scale_factor
        
        return reconstructed
        
    def write_lammps_trajectory(self, filename: str, displacements: np.ndarray,
                                start_time: int=0, timestep: int=1) -> None:
        n_frames, n_atoms, _ = displacements.shape
        ref_positions = np.mean(self.traj.positions, axis=0)
        box = self.traj.box
        
        xy, xz, yz = box.tilts
        xlo=0.0
        xhi=box.lengths[0]
        ylo=0.0
        yhi=box.lengths[1]
        zlo=0.0
        zhi=box.lengths[2]
        
        xlo_bound = xlo+min(0.0,xy,xz,xy+xz)
        xhi_bound = xhi+max(0.0,xy,xz,xy+xz)
        ylo_bound = ylo+min(0.0,yz)
        yhi_bound = yhi+max(0.0,yz)
        
        logger.info(f"Writing reconstructed trajectory to {filename}")
        try:
            with open(filename, 'w') as f:
                for t in tqdm(range(n_frames), desc="Writing trajectory", unit="frame"):
                    f.write("ITEM: TIMESTEP\n")
                    f.write(f"{int(start_time + t*timestep)}\n")
                    f.write("ITEM: NUMBER OF ATOMS\n")
                    f.write(f"{n_atoms}\n")
                    f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                    f.write(f"{xlo_bound:.6f} {xhi_bound:.6f} {xy:.6f}\n")
                    f.write(f"{ylo_bound:.6f} {yhi_bound:.6f} {xz:.6f}\n")
                    f.write(f"{zlo:.6f} {zhi:.6f} {yz:.6f}\n")
                    
                    positions = ref_positions + displacements[t]
                    
                    f.write("ITEM: ATOMS id type x y z\n")
                    for i in range(n_atoms):
                        x_,y_,z_= positions[i]
                        atype = self.traj.types[i]
                        f.write(f"{i+1} {atype} {x_:.6f} {y_:.6f} {z_:.6f}\n")
            logger.info("Reconstructed trajectory successfully written.")
        except Exception as e:
            logger.error(f"Failed to write LAMMPS trajectory: {e}")
            raise

def main():
    import sys
    parser = argparse.ArgumentParser(description='Spectral Displacement (SD) Analysis Tool')
    parser.add_argument('trajectory', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output dir')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = {}
    if args.config:
        try:
            with open(args.config,'r') as fh:
                config = yaml.safe_load(fh) or {}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    dt = config.get('dt',0.005)
    nx = config.get('nx',60)
    ny = config.get('ny',60)
    nz = config.get('nz',1)
    direction = config.get('direction','x')
    bz_coverage = config.get('bz_coverage',1.0)
    n_kpoints = config.get('n_kpoints',60)
    wmin = config.get('wmin',0)
    wmax = config.get('wmax',50)
    max_freq = config.get('max_freq',None)
    kmin = config.get('kmin',None)
    kmax = config.get('kmax',None)
    amplitude = config.get('amplitude',1.0)
    lattice_parameter = config.get('lattice_parameter',None)
    do_filtering = config.get('do_filtering',True)
    do_reconstruction = config.get('do_reconstruction',True)
    use_velocities = config.get('use_velocities',False)
    save_npy = config.get('save_npy',False)
    
    try:
        loader = TrajectoryLoader(args.trajectory, dt=dt)
        traj = loader.load()
        
        if save_npy:
            logger.info("Saving trajectory data as numpy arrays...")
            loader.save_numpy_arrays(traj)
        
        calc = SDCalculator(traj,nx,ny,nz,use_velocities=use_velocities)
        k_points, k_vectors = calc.get_k_path(direction,bz_coverage,n_kpoints,
                                              lattice_parameter=lattice_parameter)
        
        logger.info("Calculating SD...")
        power_spectrum, freqs = calc.calculate_sd(k_points,k_vectors)
        sd = power_spectrum
        
        full_intensity = np.sum(sd, axis=-1)
        max_int = np.max(np.abs(full_intensity))  # safer to consider abs if complex
        
        data_type = 'vel' if use_velocities else 'disp'
        global_plot = out_dir / f'sd_global_{data_type}.png'
        filtered_plot = out_dir / f'sd_filtered_{data_type}.png'
        out_traj = out_dir / f'reconstructed_{data_type}.lammpstrj'
        
        # Plot unfiltered
        calc.plot_sed(
            sd, freqs, k_points,
            output=str(global_plot),
            global_max_intensity=max_int,
            max_freq=max_freq
        )
        
        filtered_sd = sd
        if do_filtering:
            logger.info("Applying filters to SD...")
            freq_range = (wmin,wmax)
            k_range = (kmin,kmax) if kmin is not None and kmax is not None else None
            filtered_sd = calc.filter_sd(sd,freqs,k_points,
                                         freq_range=freq_range,
                                         k_range=k_range)
            calc.plot_sed(
                filtered_sd, freqs, k_points,
                output=str(filtered_plot),
                global_max_intensity=max_int,
                max_freq=max_freq,
                vmin=np.sqrt(1e-6),
                vmax=np.sqrt(1.0),
                highlight_region={'freq_range':freq_range,'k_range':k_range}
            )
        
        if do_reconstruction:
            logger.info("Reconstructing atomic displacements...")
            recon = TrajectoryReconstructor(traj,calc)
            disp = recon.reconstruct_mode(filtered_sd,freqs,k_points,k_vectors,
                                          desired_amplitude=amplitude)
            recon.write_lammps_trajectory(str(out_traj),disp)
        
        if save_npy:
            logger.info("Saving SD data as numpy arrays...")
            loader.save_numpy_arrays(traj, sd=sd, freqs=freqs, use_velocities=use_velocities)
        
        logger.info("SD analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)

if __name__=="__main__":
    main()
