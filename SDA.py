import numpy as np
from pathlib import Path
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Union
import logging
import argparse
import yaml
from tqdm import tqdm
import os

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors
except ImportError as e:
    logging.error(f"Matplotlib import failed: {e}")
    raise

try:
    import ovito
    from ovito.io import import_file
    from ovito.modifiers import UnwrapTrajectoriesModifier
except ImportError as e:
    logging.error(f"OVITO import failed: {e}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')


# ------------------ Data Structures ------------------
@dataclass
class Box:
    lengths: np.ndarray
    tilts: np.ndarray
    matrix: np.ndarray

    @classmethod
    def from_ovito(cls, cell) -> 'Box':
        matrix = cell.matrix.copy().astype(np.float32)
        lengths = np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2]], dtype=np.float32)
        tilts = np.array([matrix[0, 1], matrix[0, 2], matrix[1, 2]], dtype=np.float32)
        return cls(lengths=lengths, tilts=tilts, matrix=matrix)


@dataclass
class Trajectory:
    positions: np.ndarray      # (n_frames, n_atoms, 3)
    velocities: np.ndarray     # (n_frames, n_atoms, 3)
    types: np.ndarray          # (n_atoms,)
    timesteps: np.ndarray      # (n_frames,)
    box: Box

    def __post_init__(self):
        if self.positions.ndim != 3:
            raise ValueError("Positions must be 3D (frames, atoms, xyz)")
        if self.velocities.ndim != 3:
            raise ValueError("Velocities must be 3D (frames, atoms, xyz)")
        if self.types.ndim != 1:
            raise ValueError("Types must be 1D")
        if self.timesteps.ndim != 1:
            raise ValueError("Timesteps must be 1D")

    @property
    def n_frames(self) -> int:
        return len(self.timesteps)

    @property
    def n_atoms(self) -> int:
        return len(self.types)


# ------------------ Trajectory Loader ------------------
class TrajectoryLoader:
    def __init__(self, filename: str, dt: float = 1.0, file_format: str = 'auto'):
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.filepath = Path(filename)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filename}")
        self.dt = dt
        valid_formats = ['auto', 'lammps', 'vasp_outcar']
        if file_format not in valid_formats:
            raise ValueError(f"Unsupported file format. Must be one of: {valid_formats}")
        self.file_format = file_format

    def _detect_file_format(self) -> str:
        if self.file_format != 'auto':
            return self.file_format
        if self.filepath.suffix.lower() == '.outcar':
            return 'vasp_outcar'
        try:
            with open(self.filepath, 'r') as f:
                first_line = f.readline().strip()
                if 'OUTCAR' in first_line or 'vasp' in first_line.lower():
                    return 'vasp_outcar'
        except:
            pass
        return 'lammps'

    def load(self) -> Trajectory:
        base_path = self.filepath.with_suffix('')
        npy_files = {
            'positions': base_path.with_suffix('.positions.npy'),
            'velocities': base_path.with_suffix('.velocities.npy'),
            'types': base_path.with_suffix('.types.npy'),
        }
        if all(f.exists() for f in npy_files.values()):
            logger.info("Found .npy files; loading trajectory.")
            try:
                positions = np.load(npy_files['positions'])
                velocities = np.load(npy_files['velocities'])
                types = np.load(npy_files['types'])
                pipeline = import_file(str(self.filepath))
                pipeline.modifiers.append(UnwrapTrajectoriesModifier())
                frame0 = pipeline.compute(0)
                box = Box.from_ovito(frame0.cell)
                n_frames = positions.shape[0]
                timesteps = np.arange(n_frames, dtype=np.float32)
                logger.info("Trajectory loaded from .npy files.")
                return Trajectory(positions, velocities, types, timesteps, box)
            except Exception as e:
                logger.warning(f"Loading .npy failed: {e}. Using OVITO.")
                return self._load_via_ovito()
        else:
            logger.info("No .npy files found; loading via OVITO.")
            return self._load_via_ovito()

    def _load_via_ovito(self) -> Trajectory:
        logger.info("Loading and unwrapping trajectory with OVITO.")
        fmt = self._detect_file_format()
        logger.info(f"Detected format: {fmt}")
        if fmt == 'vasp_outcar':
            pipeline = import_file(str(self.filepath), 
                                   columns=["Particle Type", "Position.X", "Position.Y", "Position.Z", 
                                            "Velocity.X", "Velocity.Y", "Velocity.Z"])
        else:
            pipeline = import_file(str(self.filepath))
        pipeline.modifiers.append(UnwrapTrajectoriesModifier())
        n_frames = pipeline.source.num_frames
        frame0 = pipeline.compute(0)
        n_atoms = len(frame0.particles.positions)
        if not hasattr(frame0.particles, 'velocities'):
            raise ValueError("No velocity data found in trajectory.")
        positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        velocities = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        for i in tqdm(range(n_frames), desc="Loading frames", unit="frame"):
            frame = pipeline.compute(i)
            positions[i] = frame.particles.positions.array.astype(np.float32)
            velocities[i] = frame.particles.velocities.array.astype(np.float32)
        types = frame0.particles.particle_types.array
        timesteps = np.arange(n_frames, dtype=np.float32)
        box = Box.from_ovito(frame0.cell)
        logger.info("Trajectory loaded via OVITO.")
        return Trajectory(positions, velocities, types, timesteps, box)

    def save_trajectory_npy(self, traj: Trajectory) -> None:
        base_path = self.filepath.parent / self.filepath.stem
        exists = all((base_path.with_suffix(suffix)).exists() 
                     for suffix in ['.positions.npy', '.velocities.npy', '.types.npy'])
        if exists:
            logger.info("Trajectory npy files exist; skipping save.")
            return
        logger.info("Saving trajectory to .npy files.")
        np.save(base_path.with_suffix('.positions.npy'), traj.positions)
        np.save(base_path.with_suffix('.velocities.npy'), traj.velocities)
        np.save(base_path.with_suffix('.types.npy'), traj.types)
        mean_positions = np.mean(traj.positions, axis=0)
        displacements = traj.positions - mean_positions[None, :, :]
        np.save(base_path.with_suffix('.mean_positions.npy'), mean_positions)
        np.save(base_path.with_suffix('.displacements.npy'), displacements)
        logger.info("Trajectory saved to .npy files.")


# ------------------ Helper Functions ------------------
def parse_direction(direction: Union[str, int, float, List[float], Dict[str, float]]) -> np.ndarray:
    if isinstance(direction, (int, float)):
        rad = np.deg2rad(direction)
        vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
    elif isinstance(direction, str):
        try:
            angle = float(direction)
            rad = np.deg2rad(angle)
            vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        except ValueError:
            mapping = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
            if direction.lower() not in mapping:
                raise ValueError(f"Unknown direction: {direction}")
            vec = np.array(mapping[direction.lower()], dtype=np.float32)
    elif isinstance(direction, (list, tuple, np.ndarray)):
        if len(direction) == 1:
            angle = float(direction[0])
            rad = np.deg2rad(angle)
            vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        elif len(direction) == 3:
            vec = np.array(direction, dtype=np.float32)
        else:
            raise ValueError("Direction must have 1 or 3 components")
    elif isinstance(direction, dict):
        if 'angle' in direction:
            angle = float(direction['angle'])
            rad = np.deg2rad(angle)
            vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        else:
            vec = np.array([direction.get('h', 0.0),
                            direction.get('k', 0.0),
                            direction.get('l', 0.0)], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported direction type: {type(direction)}")
    if np.linalg.norm(vec) < 1e-10:
        raise ValueError("Zero direction vector not allowed")
    return vec


# ------------------ SD Calculation ------------------
class SDCalculator:
    def __init__(self,
                 traj: Trajectory,
                 nx: int,
                 ny: int,
                 nz: int,
                 dt_ps: float,
                 use_velocities: bool = False):
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("System dimensions must be positive")
        self.traj = traj
        self.use_velocities = use_velocities
        self.dt_ps = dt_ps
        cell_mat = self.traj.box.matrix.astype(np.float32)
        self.a1 = cell_mat[:, 0] / float(nx)
        self.a2 = cell_mat[:, 1] / float(ny)
        self.a3 = cell_mat[:, 2] / float(nz)
        volume = np.dot(self.a1, np.cross(self.a2, self.a3))
        b1 = 2 * np.pi * np.cross(self.a2, self.a3) / volume
        b2 = 2 * np.pi * np.cross(self.a3, self.a1) / volume
        b3 = 2 * np.pi * np.cross(self.a1, self.a2) / volume
        self.recip_vectors = np.vstack([b1, b2, b3]).astype(np.float32)

    def get_k_path(self,
                   direction: Union[str, int, float, List[float], np.ndarray],
                   bz_coverage: float,
                   n_k: int,
                   lattice_parameter: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        dir_vector = parse_direction(direction)
        dir_vector /= np.linalg.norm(dir_vector)
        if lattice_parameter is None:
            lattice_parameter = np.linalg.norm(self.a1)
            logger.info(f"Using lattice parameter: {lattice_parameter:.3f} Å")
        k_max = bz_coverage * (2 * np.pi / lattice_parameter)
        k_points = np.linspace(0, k_max, n_k, dtype=np.float32)
        k_vectors = np.outer(k_points, dir_vector).astype(np.float32)
        return k_points, k_vectors

    def calculate_sd(self, k_points: np.ndarray, k_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_t = self.traj.n_frames
        mean_positions = np.mean(self.traj.positions, axis=0)
        sed = np.zeros((n_t, len(k_points), 3), dtype=np.complex64)
        data_array = self.traj.velocities if self.use_velocities else (self.traj.positions - mean_positions)
        for ik, kvec in enumerate(tqdm(k_vectors, desc="Processing k-points", unit="k-point")):
            phase = np.exp(1j * np.dot(mean_positions, kvec))
            for alpha in range(3):
                sed[:, ik, alpha] = np.sum(data_array[..., alpha] * phase, axis=1)
        dt_s = self.dt_ps * 1e-12
        sed_w = np.fft.fft(sed, axis=0)
        freqs = np.fft.fftfreq(n_t, d=dt_s) * 1e-12
        return sed_w, freqs

    def plot_sed(self,
                 sed: np.ndarray,
                 freqs: np.ndarray,
                 k_points: np.ndarray,
                 output: str,
                 direction_label: str = '',
                 cmap: str = 'inferno',
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 global_max_intensity: Optional[float] = None,
                 highlight_region: Optional[Dict[str, Tuple[float, float]]] = None,
                 max_freq: Optional[float] = None) -> None:
        try:
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            sed = sed[pos_mask]
            intensity = np.abs(sed).sum(axis=-1).real.astype(np.float32)
            k_mesh, f_mesh = np.meshgrid(k_points, freqs)
            if global_max_intensity is not None and global_max_intensity > 0:
                intensity /= global_max_intensity
            else:
                m = np.max(intensity)
                if m > 0:
                    intensity /= m
            sqrt_intensity = np.sqrt(intensity + 1e-20)
            if vmin is None:
                vmin = np.percentile(sqrt_intensity[sqrt_intensity > 0], 1)
            if vmax is None:
                vmax = np.percentile(sqrt_intensity[sqrt_intensity > 0], 99)
            plt.figure(figsize=(10, 8))
            pcm = plt.pcolormesh(k_mesh, f_mesh, sqrt_intensity,
                                 shading='gouraud', cmap=cmap,
                                 vmin=vmin, vmax=vmax)
            if highlight_region:
                fr = highlight_region.get('freq_range')
                kr = highlight_region.get('k_range')
                if fr and kr:
                    f_min, f_max = fr
                    k_min, k_max = kr
                    rect = plt.Rectangle((k_min, f_min), k_max - k_min, f_max - f_min,
                                         fill=False, edgecolor='white', linestyle='--', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(k_max + 0.05, 0.5 * (f_max + f_min),
                             f'Selected\nRegion\n{f_min}-{f_max} THz\n{k_min}-{k_max} (2π/Å)',
                             color='white', va='center', fontsize=8)
            plt.xlabel('k (2π/Å)')
            plt.ylabel('Frequency (THz)')
            plt.ylim(0, max_freq if max_freq and max_freq > 0 else np.max(freqs))
            plt.title(f'SED {direction_label}')
            plt.colorbar(pcm, label='√Intensity (arb. units)')
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SED plot saved: {output}")
        except Exception as e:
            logger.error(f"SED plot error: {e}")
            raise


# ------------------ Time-Domain Filtering ------------------
class TimeDomainFilter:
    def __init__(self, dt_ps: float):
        if dt_ps <= 0:
            raise ValueError("dt must be positive.")
        self.dt_ps = dt_ps
        self.dt_s = dt_ps * 1e-12

    def filter_in_frequency(self,
                            data: np.ndarray,
                            w_min: float,
                            w_max: float) -> np.ndarray:
        n_frames, n_atoms, _ = data.shape
        freqs = np.fft.fftfreq(n_frames, d=self.dt_s) * 1e-12
        w_center = 0.5 * (w_min + w_max)
        w_sigma = (w_max - w_min) / 6.0 if w_max > w_min else 0.0
        logger.info(f"Gaussian filter: center={w_center:.2f} THz, sigma={w_sigma:.2f} THz")
        if w_sigma < 1e-14:
            logger.warning("Filter width too small; returning original data.")
            return data.copy()
        freq_window = np.exp(-0.5 * ((freqs - w_center) / w_sigma) ** 2)
        filtered_data = np.zeros_like(data, dtype=np.float32)
        for i_atom in tqdm(range(n_atoms), desc="Filtering atoms", unit="atom"):
            for alpha in range(3):
                ts = data[:, i_atom, alpha]
                fft_vals = np.fft.fft(ts)
                fft_filtered = fft_vals * freq_window
                filtered_data[:, i_atom, alpha] = np.fft.ifft(fft_filtered).real
        return filtered_data


def write_filtered_trajectory(filename: str,
                              ref_positions: np.ndarray,
                              box: Box,
                              filtered_data: np.ndarray,
                              types: np.ndarray,
                              dt_ps: float,
                              start_time_ps: float = 0.0):
    n_frames, n_atoms, _ = filtered_data.shape
    xy, xz, yz = box.tilts
    xlo, xhi = 0.0, box.lengths[0]
    ylo, yhi = 0.0, box.lengths[1]
    zlo, zhi = 0.0, box.lengths[2]
    xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
    xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
    ylo_bound = ylo + min(0.0, yz)
    yhi_bound = yhi + max(0.0, yz)
    logger.info(f"Writing filtered trajectory to {filename}")
    with open(filename, 'w') as f:
        for frame_idx in tqdm(range(n_frames), desc="Writing frames", unit="frame"):
            current_time_ps = start_time_ps + frame_idx * dt_ps
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{current_time_ps:.6f}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{n_atoms}\n")
            f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            f.write(f"{xlo_bound:.6f} {xhi_bound:.6f} {xy:.6f}\n")
            f.write(f"{ylo_bound:.6f} {yhi_bound:.6f} {xz:.6f}\n")
            f.write(f"{zlo:.6f} {zhi:.6f} {yz:.6f}\n")
            coords = ref_positions + filtered_data[frame_idx]
            f.write("ITEM: ATOMS id type x y z\n")
            for i_atom in range(n_atoms):
                x, y, z = coords[i_atom]
                atype = types[i_atom]
                f.write(f"{i_atom+1} {atype} {x:.6f} {y:.6f} {z:.6f}\n")


# ------------------ 3D Dispersion Plotting ------------------
def gather_3d_data(k_vectors_list: List[np.ndarray],
                   freqs_list: List[np.ndarray],
                   sed_list: List[np.ndarray],
                   intensity_threshold: float = 0.01):
    kx_vals, ky_vals, freq_vals, amp_vals = [], [], [], []
    for freqs, kvecs, sed in zip(freqs_list, k_vectors_list, sed_list):
        intensity_3d = np.abs(sed).sum(axis=-1)
        for i in range(len(freqs)):
            for j in range(len(kvecs)):
                kx_vals.append(kvecs[j][0])
                ky_vals.append(kvecs[j][1])
                freq_vals.append(freqs[i])
                amp_vals.append(intensity_3d[i, j])
    return (np.array(kx_vals, dtype=np.float32),
            np.array(ky_vals, dtype=np.float32),
            np.array(freq_vals, dtype=np.float32),
            np.array(amp_vals, dtype=np.float32))


def plot_3d_dispersion(kx_vals: np.ndarray,
                       ky_vals: np.ndarray,
                       freq_vals: np.ndarray,
                       amp_vals: np.ndarray,
                       output_path: str):
    if len(kx_vals) == 0:
        logger.warning("No 3D data points to plot.")
        return
    amp_vals[amp_vals < 1e-20] = 1e-20
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        kx_vals, ky_vals, freq_vals,
        c=np.log(amp_vals),
        cmap='plasma',
        alpha=0.7,
        marker='o',
        s=10,
        edgecolors='none',
        norm=mcolors.Normalize(vmin=np.log(amp_vals.min()),
                               vmax=np.log(amp_vals.max()))
    )
    ax.set_xlabel(r'$k_x$ (2$\pi$/Å)')
    ax.set_ylabel(r'$k_y$ (2$\pi$/Å)')
    ax.set_zlabel('Frequency (THz)')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('log(Amplitude)')
    plt.title('3D Dispersion Visualization')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"3D dispersion plot saved: {output_path}")


# ------------------ New: Chiral SED Functions ------------------
def SEDphase(Z1, Z2, angleRange="C"):
    if angleRange == "C":
        t1 = np.arctan2(Z1.real, Z1.imag)
        t2 = np.arctan2(Z2.real, Z2.imag)
        dt = t1 - t2
        dt[dt > np.pi] -= 2 * np.pi
        dt[dt < -np.pi] += 2 * np.pi
        Q2 = dt > (np.pi / 2)
        dt[Q2] = -dt[Q2] + np.pi
        Q3 = dt < (-np.pi / 2)
        dt[Q3] = -dt[Q3] - np.pi
        return dt
    else:
        nw, nk = Z1.shape
        out = np.zeros((nw, nk))
        for i in range(nw):
            for j in range(nk):
                v1 = np.array([Z1[i, j].real, Z1[i, j].imag])
                v2 = np.array([Z2[i, j].real, Z2[i, j].imag])
                m1 = np.linalg.norm(v1)
                m2 = np.linalg.norm(v2)
                if angleRange == "A":
                    dot_val = np.clip(np.dot(v1, v2) / (m1 * m2), -1, 1)
                    angle = np.arccos(dot_val)
                elif angleRange == "B":
                    cross_val = np.clip(np.cross(v1, v2) / (m1 * m2), -1, 1)
                    angle = np.arcsin(cross_val)
                else:
                    angle = 0.0
                out[i, j] = angle
        return out


def plot_chiral_sed(phase_dict: Dict[str, np.ndarray],
                    freqs: np.ndarray,
                    k_points: np.ndarray,
                    output: str,
                    direction_label: str = '',
                    vmin: float = -np.pi / 2,
                    vmax: float = np.pi / 2,
                    cmap: str = 'twilight'):
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    k_mesh, f_mesh = np.meshgrid(k_points, freqs)
    pair_labels = {"phase_0_1": "v0 vs v1", "phase_0_2": "v0 vs v2", "phase_1_2": "v1 vs v2"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    pcm_list = []
    for ax, key in zip(axs, ["phase_0_1", "phase_0_2", "phase_1_2"]):
        phase = phase_dict[key][pos_mask]
        pcm = ax.pcolormesh(k_mesh, f_mesh, phase,
                            shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{pair_labels[key]} {direction_label}")
        ax.set_xlabel('k (2π/Å)')
        pcm_list.append(pcm)
    axs[0].set_ylabel('Frequency (THz)')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pcm_list[0], cax=cbar_ax, label='Phase difference (rad)')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Chiral SED plot saved: {output}")


def gather_3d_chiral_data(k_vectors_list: List[np.ndarray],
                          freqs_list: List[np.ndarray],
                          phase_list: List[np.ndarray]):
    kx_vals, ky_vals, freq_vals, phase_vals = [], [], [], []
    for freqs, kvecs, phase in zip(freqs_list, k_vectors_list, phase_list):
        n_freq, n_k = phase.shape
        for i in range(n_freq):
            for j in range(n_k):
                kx_vals.append(kvecs[j][0])
                ky_vals.append(kvecs[j][1])
                freq_vals.append(freqs[i])
                phase_vals.append(phase[i, j])
    return (np.array(kx_vals, dtype=np.float32),
            np.array(ky_vals, dtype=np.float32),
            np.array(freq_vals, dtype=np.float32),
            np.array(phase_vals, dtype=np.float32))


def plot_3d_chiral_dispersion(kx_vals: np.ndarray,
                              ky_vals: np.ndarray,
                              freq_vals: np.ndarray,
                              phase_vals: np.ndarray,
                              output_path: str):
    if len(kx_vals) == 0:
        logger.warning("No 3D chiral data to plot.")
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        kx_vals, ky_vals, freq_vals,
        c=phase_vals,
        cmap='coolwarm',
        alpha=0.7,
        marker='o',
        s=10,
        edgecolors='none',
        vmin=-np.pi / 2, vmax=np.pi / 2
    )
    ax.set_xlabel(r'$k_x$ (2$\pi$/Å)')
    ax.set_ylabel(r'$k_y$ (2$\pi$/Å)')
    ax.set_zlabel('Frequency (THz)')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Phase difference (rad)')
    plt.title('3D Chiral Dispersion')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"3D chiral dispersion plot saved: {output_path}")


# ------------------ Main Routine ------------------
def main():
    parser = argparse.ArgumentParser(
        description='Spectral Displacement Analysis Tool with optional chiral SED'
    )
    parser.add_argument('trajectory', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--chiral', action='store_true',
                        help='Compute only chiral SED (no regular SED) for all directions')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default configuration
    config = {
        'dt': 0.005,
        'nx': 60,
        'ny': 60,
        'nz': 1,
        'directions': None,    # e.g. [0, 45, 90]
        'n_kpoints': 60,
        'bz_coverage': 1.0,
        'max_freq': 50,
        'wmin': 0,
        'wmax': 50,
        'kmin': None,
        'kmax': None,
        'amplitude': 1.0,
        'lattice_parameter': None,
        'do_filtering': False,
        'do_reconstruction': False,
        'use_velocities': True,
        'save_npy': True,
        '3D_Dispersion': False,
        'chiral': args.chiral
    }
    if args.config:
        try:
            with open(args.config, 'r') as fh:
                user_config = yaml.safe_load(fh) or {}
                config.update(user_config)
        except Exception as e:
            logger.warning(f"Config load failed: {e}")

    directions = config['directions'] if config.get('directions') is not None else [0]

    # For accumulating chiral data across all directions
    all_chiral_data = {}
    # For 3D chiral dispersion plotting (using one polarization pair as example)
    all_kvecs = []
    all_freqs = []
    all_chiral_phase = []

    try:
        # Load trajectory
        loader = TrajectoryLoader(args.trajectory, dt=config['dt'])
        traj = loader.load()
        if config['save_npy']:
            loader.save_trajectory_npy(traj)

        sd_calc = SDCalculator(
            traj=traj,
            nx=config['nx'],
            ny=config['ny'],
            nz=config['nz'],
            dt_ps=config['dt'],
            use_velocities=config['use_velocities']
        )

        for i_dir, angle in enumerate(directions, start=1):
            angle_str = f"{float(angle):.1f}"
            logger.info(f"Processing chiral SED for angle: {angle_str}°")
            k_points, k_vectors = sd_calc.get_k_path(
                direction=angle,
                bz_coverage=config['bz_coverage'],
                n_k=config['n_kpoints'],
                lattice_parameter=config['lattice_parameter']
            )
            sed, freqs = sd_calc.calculate_sd(k_points, k_vectors)
            phase_dict = {
                "phase_0_1": SEDphase(sed[:, :, 0], sed[:, :, 1], angleRange="C"),
                "phase_0_2": SEDphase(sed[:, :, 0], sed[:, :, 2], angleRange="C"),
                "phase_1_2": SEDphase(sed[:, :, 1], sed[:, :, 2], angleRange="C")
            }
            # Accumulate data for this direction into the large chiral.npz file.
            all_chiral_data[f"dir_{i_dir}"] = {
                "angle": angle,
                "k_points": k_points,
                "freqs": freqs,
                "phase_0_1": phase_dict["phase_0_1"],
                "phase_0_2": phase_dict["phase_0_2"],
                "phase_1_2": phase_dict["phase_1_2"]
            }
            # Plot chiral SED for this direction.
            plot_file = out_dir / f"{i_dir:03d}_chiral_sed_{angle_str}deg.png"
            plot_chiral_sed(phase_dict, freqs, k_points, str(plot_file),
                            direction_label=f"(angle: {angle_str}°)")
            # Collect data for 3D chiral dispersion (using phase_0_1 as example).
            all_kvecs.append(k_vectors)
            all_freqs.append(freqs)
            all_chiral_phase.append(phase_dict["phase_0_1"])

        # Save one large chiral.npz containing all directions.
        chiral_all_path = out_dir / "chiral.npz"
        # Create a flat dictionary with keys like "dir_1_angle", "dir_1_k_points", etc.
        chiral_data_flat = {}
        for key, data in all_chiral_data.items():
            chiral_data_flat[f"{key}_angle"] = data["angle"]
            chiral_data_flat[f"{key}_k_points"] = data["k_points"]
            chiral_data_flat[f"{key}_freqs"] = data["freqs"]
            chiral_data_flat[f"{key}_phase_0_1"] = data["phase_0_1"]
            chiral_data_flat[f"{key}_phase_0_2"] = data["phase_0_2"]
            chiral_data_flat[f"{key}_phase_1_2"] = data["phase_1_2"]
        np.savez(chiral_all_path, **chiral_data_flat)
        logger.info(f"All chiral SED data saved in one file: {chiral_all_path}")

        # 3D dispersion plot for chiral data (if requested)
        if config.get('3D_Dispersion', False):
            logger.info("Generating 3D chiral dispersion plot.")
            kx, ky, freq_vals, phase_vals = gather_3d_chiral_data(
                k_vectors_list=all_kvecs,
                freqs_list=all_freqs,
                phase_list=all_chiral_phase
            )
            npz_path = out_dir / "3d_chiral_dispersion_data.npz"
            np.savez(npz_path, kx=kx, ky=ky, freq=freq_vals, phase=phase_vals)
            logger.info(f"3D chiral dispersion data saved: {npz_path}")
            plot_3d_chiral_dispersion(
                kx, ky, freq_vals, phase_vals,
                output_path=str(out_dir / "3d_chiral_dispersion.png")
            )

        if config['do_filtering']:
            logger.info("Applying time-domain frequency filter.")
            filter_obj = TimeDomainFilter(dt_ps=config['dt'])
            if config['use_velocities']:
                data_array = traj.velocities
                logger.info("Filtering velocities.")
            else:
                mean_positions = np.mean(traj.positions, axis=0)
                data_array = traj.positions - mean_positions
                logger.info("Filtering displacements.")
            filtered_data = filter_obj.filter_in_frequency(
                data=data_array,
                w_min=config['wmin'],
                w_max=config['wmax']
            )
            if config['amplitude'] != 1.0:
                rms = np.sqrt(np.mean(filtered_data**2))
                if rms > 1e-14:
                    factor = config['amplitude'] / rms
                    filtered_data *= factor
                    logger.info(f"Scaled filtered data by factor {factor:.3f}")
            if config['do_reconstruction']:
                logger.info("Writing filtered trajectory.")
                ref = traj.positions[0] if config['use_velocities'] else np.mean(traj.positions, axis=0)
                out_file = out_dir / 'filtered_time_domain.lammpstrj'
                write_filtered_trajectory(
                    filename=str(out_file),
                    ref_positions=ref,
                    box=traj.box,
                    filtered_data=filtered_data,
                    types=traj.types,
                    dt_ps=config['dt']
                )

        logger.info("SD analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
