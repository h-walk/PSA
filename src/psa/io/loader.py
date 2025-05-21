"""
Trajectory loading and processing module.
"""
import numpy as np
from pathlib import Path
import logging
from typing import Optional
from tqdm import tqdm

from ..core.trajectory import Trajectory

try:
    from ovito.io import import_file
    from ovito.modifiers import UnwrapTrajectoriesModifier
except ImportError as e:
    logging.error(f"OVITO import failed: {e}")
    raise

logger = logging.getLogger(__name__)

class TrajectoryLoader:
    def __init__(self, filename: str, dt: float = 1.0, file_format: str = 'auto'):
        if dt <= 0:
            raise ValueError("dt (timestep size) must be positive.")
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
        return 'lammps'

    def load(self) -> Trajectory:
        cache_stem = self.filepath.parent / self.filepath.stem
        npy_files = {
            'positions': cache_stem.with_suffix('.positions.npy'),
            'velocities': cache_stem.with_suffix('.velocities.npy'),
            'types': cache_stem.with_suffix('.types.npy'),
            'box_matrix': cache_stem.with_suffix('.box_matrix.npy')
        }
        if all(f.exists() for f in npy_files.values()):
            logger.info(f"Loading trajectory from cached .npy files for {self.filepath.name}.")
            try:
                pos = np.load(npy_files['positions'])
                vel = np.load(npy_files['velocities'])
                atom_types = np.load(npy_files['types'])
                box_mat = np.load(npy_files['box_matrix'])

                if box_mat.shape != (3,3):
                    raise ValueError(f"Cached box_matrix has shape {box_mat.shape}, expected (3,3).")

                box_len = np.array([box_mat[0,0], box_mat[1,1], box_mat[2,2]], dtype=np.float32)
                box_tilt = np.array([box_mat[0,1], box_mat[0,2], box_mat[1,2]], dtype=np.float32)
                
                n_frames = pos.shape[0]
                ts = np.arange(n_frames, dtype=np.float32) * self.dt
                return Trajectory(pos, vel, atom_types, ts, 
                                box_matrix=box_mat, box_lengths=box_len, box_tilts=box_tilt,
                                dt_ps=self.dt)
            except Exception as e:
                logger.warning(f"Loading .npy cache failed: {e}. Falling back to OVITO.")

        logger.info(f"No complete .npy cache for {self.filepath.name}; loading via OVITO.")
        return self._load_via_ovito()

    def _load_via_ovito(self) -> Trajectory:
        logger.info(f"Loading/unwrapping '{self.filepath.name}' with OVITO.")
        ovito_fmt = None
        detected_fmt = self._detect_file_format()
        if detected_fmt == 'lammps':
            ovito_fmt = 'lammps/dump'
        elif detected_fmt == 'vasp_outcar':
            ovito_fmt = 'vasp/outcar'
        
        logger.debug(f"OVITO load format: {ovito_fmt or 'auto-detected'}")

        pipeline = import_file(str(self.filepath), input_format=ovito_fmt)
        pipeline.modifiers.append(UnwrapTrajectoriesModifier())

        n_frames = pipeline.source.num_frames
        if n_frames == 0:
            raise ValueError("OVITO: 0 frames in trajectory.")

        frame0_data = pipeline.compute(0)
        if not (frame0_data and hasattr(frame0_data, 'cell') and frame0_data.cell):
             raise ValueError("OVITO: Could not read cell data from frame 0.")
        if not (hasattr(frame0_data, 'particles') and frame0_data.particles):
             raise ValueError("OVITO: Could not read particle data from frame 0.")
        
        n_atoms = len(frame0_data.particles.positions) if hasattr(frame0_data.particles, 'positions') and frame0_data.particles.positions is not None else 0
        if n_atoms == 0:
            raise ValueError("OVITO: 0 atoms in frame 0.")

        has_vel = hasattr(frame0_data.particles, 'velocities') and frame0_data.particles.velocities is not None
        if not has_vel:
            logger.warning("OVITO: No velocity data found. Velocities set to zero.")

        pos_all = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        vel_all = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        
        h_matrix = np.array(frame0_data.cell.matrix, dtype=np.float32)[:3,:3]
        box_len = np.array([h_matrix[0,0], h_matrix[1,1], h_matrix[2,2]], dtype=np.float32)
        box_tilt = np.array([h_matrix[0,1], h_matrix[0,2], h_matrix[1,2]], dtype=np.float32)
        
        for i in tqdm(range(n_frames), desc=f"Loading OVITO frames from {self.filepath.name}", unit="fr"):
            frame_data = pipeline.compute(i)
            if not (frame_data and hasattr(frame_data, 'particles') and frame_data.particles):
                logger.error(f"OVITO: Could not compute frame {i}. Data will be zero.")
                continue
            
            if hasattr(frame_data.particles, 'positions') and frame_data.particles.positions is not None:
                 frame_pos = np.array(frame_data.particles.positions, dtype=np.float32)
                 if frame_pos.shape == (n_atoms, 3): 
                     pos_all[i] = frame_pos
                 else: 
                     logger.warning(f"OVITO: Pos shape mismatch frame {i}. Expected ({n_atoms},3), got {frame_pos.shape}.")
            else: 
                logger.warning(f"OVITO: No position data frame {i}.")

            if has_vel and hasattr(frame_data.particles, 'velocities') and frame_data.particles.velocities is not None:
                frame_vel = np.array(frame_data.particles.velocities, dtype=np.float32)
                if frame_vel.shape == (n_atoms, 3): 
                    vel_all[i] = frame_vel
                else: 
                    logger.warning(f"OVITO: Vel shape mismatch frame {i}. Expected ({n_atoms},3), got {frame_vel.shape}.")

        types_data = frame0_data.particles.particle_types if hasattr(frame0_data.particles, 'particle_types') and frame0_data.particles.particle_types is not None else None
        if types_data is not None and len(types_data) == n_atoms:
            atom_types_arr = np.array(types_data, dtype=np.int32)
        else:
            logger.warning(f"OVITO: Particle types missing/mismatched (expected {n_atoms}). Defaulting types to 1.")
            atom_types_arr = np.ones(n_atoms, dtype=np.int32)
            
        ts_arr = np.arange(n_frames, dtype=np.float32) * self.dt
        logger.info(f"Trajectory '{self.filepath.name}' loaded via OVITO: {n_frames} frames, {n_atoms} atoms.")
        
        # Create Trajectory object from OVITO data
        trajectory_from_ovito = Trajectory(pos_all, vel_all, atom_types_arr, ts_arr,
                                         box_matrix=h_matrix, box_lengths=box_len, box_tilts=box_tilt,
                                         dt_ps=self.dt)
        
        # Save the freshly loaded trajectory to .npy cache for next time
        try:
            self.save_trajectory_npy(trajectory_from_ovito)
        except Exception as e:
            logger.warning(f"Failed to save .npy cache for {self.filepath.name}: {e}")
            
        return trajectory_from_ovito

    def save_trajectory_npy(self, traj: Trajectory) -> None:
        cache_stem = self.filepath.parent / self.filepath.stem
        npy_files = {
            'positions': cache_stem.with_suffix('.positions.npy'),
            'velocities': cache_stem.with_suffix('.velocities.npy'),
            'types': cache_stem.with_suffix('.types.npy'),
            'box_matrix': cache_stem.with_suffix('.box_matrix.npy')
        }
        if all(f.exists() for f in npy_files.values()):
            logger.info(f".npy cache for {self.filepath.name} exists; skipping save.")
            return

        logger.info(f"Saving trajectory '{self.filepath.name}' to .npy (stem: {cache_stem.name}).")
        cache_stem.parent.mkdir(parents=True, exist_ok=True)

        np.save(npy_files['positions'], traj.positions)
        np.save(npy_files['velocities'], traj.velocities)
        np.save(npy_files['types'], traj.types)
        np.save(npy_files['box_matrix'], traj.box_matrix)

        mean_pos = np.mean(traj.positions, axis=0)
        disp = traj.positions - mean_pos[None, :, :] 
        np.save(cache_stem.with_suffix('.mean_positions.npy'), mean_pos)
        np.save(cache_stem.with_suffix('.displacements.npy'), disp)
        logger.info(f"Trajectory data for {self.filepath.name} saved to .npy.") 