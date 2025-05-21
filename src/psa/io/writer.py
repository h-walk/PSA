"""
Trajectory writing module for PSA.

This module provides functionality for writing trajectory data to various file formats
and saving analysis results.
"""
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Union, Dict, Any
import json
import yaml

from ..core.trajectory import Trajectory
from ..core.sed import SED

logger = logging.getLogger(__name__)

class TrajectoryWriter:
    """Class for writing trajectory data and analysis results."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the trajectory writer.
        
        Args:
            output_dir: Directory to write output files to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_sed_data(self, sed: SED, filename: Optional[str] = None) -> None:
        """
        Save SED data to a .npz file.
        
        Args:
            sed: SED object to save
            filename: Optional custom filename (default: 'sed_data.npz')
        """
        if filename is None:
            filename = 'sed_data.npz'
        filepath = self.output_dir / filename
        
        logger.info(f"Saving SED data to {filepath}")
        np.savez(
            filepath,
            k_points=sed.k_points,
            freqs=sed.freqs,
            sed=sed.sed,
            k_vectors=sed.k_vectors
        )
        if sed.phase is not None:
            np.savez_compressed(filepath.with_suffix('.phase.npz'), phase=sed.phase)
        
    def save_trajectory_data(self, traj: Trajectory, filename: Optional[str] = None) -> None:
        """
        Save trajectory data to a .npz file.
        
        Args:
            traj: Trajectory object to save
            filename: Optional custom filename (default: 'trajectory_data.npz')
        """
        if filename is None:
            filename = 'trajectory_data.npz'
        filepath = self.output_dir / filename
        
        logger.info(f"Saving trajectory data to {filepath}")
        np.savez(
            filepath,
            positions=traj.positions,
            velocities=traj.velocities,
            types=traj.types,
            timesteps=traj.timesteps,
            box_matrix=traj.box_matrix,
            box_lengths=traj.box_lengths,
            box_tilts=traj.box_tilts
        )
        
    def save_config(self, config: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Save configuration data to a YAML file.
        
        Args:
            config: Configuration dictionary to save
            filename: Optional custom filename (default: 'config.yaml')
        """
        if filename is None:
            filename = 'config.yaml'
        filepath = self.output_dir / filename
        
        logger.info(f"Saving configuration to {filepath}")
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def save_analysis_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Save analysis results to a JSON file.
        
        Args:
            results: Analysis results dictionary to save
            filename: Optional custom filename (default: 'analysis_results.json')
        """
        if filename is None:
            filename = 'analysis_results.json'
        filepath = self.output_dir / filename
        
        logger.info(f"Saving analysis results to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
            
    def save_plot(self, fig, filename: str) -> None:
        """
        Save a matplotlib figure to a file.
        
        Args:
            fig: Matplotlib figure to save
            filename: Filename to save the figure as
        """
        filepath = self.output_dir / filename
        logger.info(f"Saving plot to {filepath}")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
    def save_log(self, log_data: str, filename: Optional[str] = None) -> None:
        """
        Save log data to a text file.
        
        Args:
            log_data: Log data to save
            filename: Optional custom filename (default: 'analysis.log')
        """
        if filename is None:
            filename = 'analysis.log'
        filepath = self.output_dir / filename
        
        logger.info(f"Saving log data to {filepath}")
        with open(filepath, 'w') as f:
            f.write(log_data) 

def out_to_qdump(filename: str, positions_tf: np.ndarray, types_tf: np.ndarray, box_matrix: np.ndarray):
    n_fr, n_at, _ = positions_tf.shape
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # LAMMPS triclinic box parameters.
    # Assumes the box origin is (0,0,0) for simplicity before tilt.
    # H is the 3x3 cell matrix (box_matrix)
    # H = [[ax, bx, cx],
    #      [ay, by, cy],
    #      [az, bz, cz]]
    #
    # Lammps uses:
    # xlo, xhi, ylo, yhi, zlo, zhi
    # xy, xz, yz (tilt factors)
    #
    # x_len = H[0,0]
    # y_len = H[1,1]
    # z_len = H[2,2]
    # xy = H[0,1]  (LAMMPS 'xy' corresponds to H[1,0] in some conventions, but ovito cell.matrix is (ax ay az; bx by bz; cx cy cz).T,
    #              so H[0,1] is correct for LAMMPS xy if H is [[L_x, xy, xz], [0, L_y, yz], [0, 0, L_z]])
    # xz = H[0,2]
    # yz = H[1,2]
    #
    # The box_matrix from OVITO (and stored in Trajectory) is typically in the upper triangular form
    # or a general form that can be converted. For LAMMPS dump:
    # x_len = box_matrix[0,0]
    # y_len = box_matrix[1,1]
    # z_len = box_matrix[2,2]
    # xy    = box_matrix[0,1] # tilt factor for x in y direction
    # xz    = box_matrix[0,2] # tilt factor for x in z direction
    # yz    = box_matrix[1,2] # tilt factor for y in z direction
    #
    # LAMMPS box bounds with tilt:
    # xlo_bound = xlo + min(0.0, xy, xz, xy+xz)
    # xhi_bound = xhi + max(0.0, xy, xz, xy+xz)
    # ylo_bound = ylo + min(0.0, yz)
    # yhi_bound = yhi + max(0.0, yz)
    # zlo_bound = zlo
    # zhi_bound = zhi
    #
    # If we define xlo=0, ylo=0, zlo=0:
    xlo = 0.0
    xhi = box_matrix[0,0]
    ylo = 0.0
    yhi = box_matrix[1,1]
    zlo = 0.0
    zhi = box_matrix[2,2]

    xy = box_matrix[0,1] 
    xz = box_matrix[0,2]
    yz = box_matrix[1,2]

    is_triclinic = not (np.isclose(xy, 0.0) and np.isclose(xz, 0.0) and np.isclose(yz, 0.0))

    if is_triclinic:
        xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
        xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
        ylo_bound = ylo + min(0.0, yz)
        yhi_bound = yhi + max(0.0, yz)
        zlo_bound = zlo
        zhi_bound = zhi
    else: # Orthogonal
        xlo_bound, xhi_bound = xlo, xhi
        ylo_bound, yhi_bound = ylo, yhi
        zlo_bound, zhi_bound = zlo, zhi
        # Ensure tilt factors are strictly zero for the "pp pp pp" format string
        xy, xz, yz = 0.0, 0.0, 0.0


    with open(filename, 'w') as f:
        for i_fr in range(n_fr):
            f.write(f"ITEM: TIMESTEP\n{i_fr}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{n_at}\n")
            if is_triclinic:
                # LAMMPS convention for triclinic box bounds with tilt factors
                f.write(f"ITEM: BOX BOUNDS xy xz yz pp pp pp\n") 
                f.write(f"{xlo_bound:.8f} {xhi_bound:.8f} {xy:.8f}\n")
                f.write(f"{ylo_bound:.8f} {yhi_bound:.8f} {xz:.8f}\n")
                f.write(f"{zlo_bound:.8f} {zhi_bound:.8f} {yz:.8f}\n")
            else:
                f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
                f.write(f"{xlo_bound:.8f} {xhi_bound:.8f}\n")
                f.write(f"{ylo_bound:.8f} {yhi_bound:.8f}\n")
                f.write(f"{zlo_bound:.8f} {zhi_bound:.8f}\n")
            
            f.write("ITEM: ATOMS id type x y z\n")
            for j_at in range(n_at):
                # Ensure atom types are integers
                f.write(f"{j_at+1} {int(types_tf[j_at])} {positions_tf[i_fr,j_at,0]:.6f} {positions_tf[i_fr,j_at,1]:.6f} {positions_tf[i_fr,j_at,2]:.6f}\n")
    logger.debug(f"Wrote iSED reconstruction to Qdump: {filename}") 