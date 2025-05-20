"""
Trajectory writing module for SDA.

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

def out_to_qdump(filename: str, positions_tf: np.ndarray, types_tf: np.ndarray, sx: float, sy: float, sz: float):
    n_fr, n_at, _ = positions_tf.shape
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        for i_fr in range(n_fr):
            f.write(f"ITEM: TIMESTEP\n{i_fr}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{n_at}\n")
            f.write(f"ITEM: BOX BOUNDS pp pp pp\n0.0 {sx}\n0.0 {sy}\n0.0 {sz}\n")
            f.write("ITEM: ATOMS id type x y z\n")
            for j_at in range(n_at):
                f.write(f"{j_at+1} {int(types_tf[j_at])} {positions_tf[i_fr,j_at,0]:.6f} {positions_tf[i_fr,j_at,1]:.6f} {positions_tf[i_fr,j_at,2]:.6f}\n")
    logger.debug(f"Wrote iSED reconstruction to Qdump: {filename}") 