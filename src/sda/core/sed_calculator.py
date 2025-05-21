"""
Core SED calculation engine.
"""
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
import logging
from pathlib import Path
from tqdm import tqdm

from .trajectory import Trajectory
from .sed import SED
from ..utils.helpers import parse_direction
from ..io.writer import out_to_qdump
from ..visualization import SEDPlotter

logger = logging.getLogger(__name__)

class SEDCalculator:
    def __init__(self, traj: Trajectory, nx: int, ny: int, nz: int, 
                 use_velocities: bool = False, dt_ps: Optional[float] = None):
        if not (nx > 0 and ny > 0 and nz > 0):
            raise ValueError("System dimensions (nx, ny, nz) must be positive.")
        self.traj = traj
        self.use_velocities = use_velocities
        
        if dt_ps is not None:
            logger.warning("Explicitly providing dt_ps to SEDCalculator is deprecated. "
                           "The timestep will be taken from the Trajectory object. "
                           "The provided dt_ps will override the Trajectory's dt_ps.")
            self.dt_ps = dt_ps
        elif hasattr(self.traj, 'dt_ps') and self.traj.dt_ps is not None:
            self.dt_ps = self.traj.dt_ps
        else:
            # This case should ideally not be reached if TrajectoryLoader always sets dt_ps
            raise ValueError("Timestep dt_ps not found in Trajectory object and not provided to SEDCalculator.")

        if self.dt_ps <= 0:
            raise ValueError("Timestep dt_ps must be positive.")

        L1, L2, L3 = self.traj.box_matrix[0,:], self.traj.box_matrix[1,:], self.traj.box_matrix[2,:]
        self.a1, self.a2, self.a3 = L1/nx, L2/ny, L3/nz
        
        if any(np.linalg.norm(v) < 1e-9 for v in [self.a1, self.a2, self.a3]):
            raise ValueError("One or more primitive vectors (a1,a2,a3) near zero. Check nx,ny,nz or box matrix.")

        vol_prim = np.abs(np.dot(self.a1, np.cross(self.a2, self.a3)))
        if np.isclose(vol_prim, 0): 
            mat_A = np.vstack([self.a1, self.a2, self.a3])
            if np.linalg.matrix_rank(mat_A) < 3 or np.isclose(np.linalg.det(mat_A),0):
                 raise ValueError(f"Primitive cell vectors coplanar/collinear; volume zero ({vol_prim:.2e}).")
            else: logger.warning(f"Primitive cell volume very small ({vol_prim:.2e}).")

        self.b1 = (2*np.pi/vol_prim) * np.cross(self.a2, self.a3)
        self.b2 = (2*np.pi/vol_prim) * np.cross(self.a3, self.a1)
        self.b3 = (2*np.pi/vol_prim) * np.cross(self.a1, self.a2)
        self.recip_vecs_prim = np.vstack([self.b1, self.b2, self.b3]).astype(np.float32)

    def _calculate_sed_for_group(self, k_vectors_3d: np.ndarray, 
                                   group_atom_indices: np.ndarray, 
                                   mean_pos_all: np.ndarray) -> np.ndarray: # Returns complex SED for the group
        """Helper to calculate complex SED for a specific group of atoms."""
        n_t = self.traj.n_frames
        
        if group_atom_indices.size == 0:
            return np.zeros((n_t, len(k_vectors_3d), 3), dtype=np.complex64)

        mean_pos_group = mean_pos_all[group_atom_indices]
        
        if self.use_velocities:
            data_ft_group = self.traj.velocities[:, group_atom_indices, :]
        else:
            data_ft_group = (self.traj.positions[:, group_atom_indices, :] - mean_pos_group[None, :, :])

        n_k_vecs = len(k_vectors_3d)
        sed_tk_pol_group = np.zeros((n_t, n_k_vecs, 3), dtype=np.complex64)

        # Optimized phase factor calculation and summation using einsum
        phase_factors_exp = np.exp(1j * np.dot(k_vectors_3d, mean_pos_group.T))

        for pol_axis in range(3):
            sed_tk_pol_group[:, :, pol_axis] = np.einsum('ta,ak->tk', data_ft_group[:, :, pol_axis], phase_factors_exp.T, optimize=True)

        sed_wk_group = np.fft.fft(sed_tk_pol_group, axis=0) / n_t if n_t > 0 else np.array([],dtype=np.complex64).reshape(0,n_k_vecs,3)
        return sed_wk_group.astype(np.complex64)

    def get_k_path(self, direction_spec: Union[str, int, float, List[float], Dict[str, float], np.ndarray],
                   bz_coverage: float, n_k: int, lat_param: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns (k_magnitudes, k_vectors_3d)
        k_dir_unit = parse_direction(direction_spec)

        if lat_param is None or lat_param <= 1e-6:
            norm_a1 = np.linalg.norm(self.a1)
            if norm_a1 > 1e-6: 
                lat_param = norm_a1
                logger.info(f"Using |a1| ({lat_param:.3f} Å) as k-path lattice parameter.")
            else: 
                raise ValueError("Invalid/small lattice_param for k-path & |a1| too small for fallback.")
        
        k_max_val = bz_coverage * (2*np.pi / lat_param)
        if n_k < 1: 
            raise ValueError("n_k (k-points) must be >= 1.")
        k_mags = np.linspace(0, k_max_val, n_k, dtype=np.float32) if n_k > 1 else np.array([0.0 if np.isclose(k_max_val,0) else k_max_val], dtype=np.float32)
        k_vecs = np.outer(k_mags, k_dir_unit).astype(np.float32)
        return k_mags, k_vecs

    def calculate(self, k_points_mags: np.ndarray, k_vectors_3d: np.ndarray,
                  basis_atom_indices: Optional[Union[List[int], List[List[int]], np.ndarray]] = None,
                  basis_atom_types: Optional[Union[List[int], List[List[int]]]] = None,
                  summation_mode: str = 'coherent'
                  ) -> Tuple[np.ndarray, np.ndarray, bool]: # Returns (sed_data, freqs_thz, is_complex)
        
        if summation_mode not in ['coherent', 'incoherent']:
            raise ValueError(f"summation_mode must be 'coherent' or 'incoherent', got {summation_mode}")

        n_t, n_atoms_tot = self.traj.n_frames, self.traj.n_atoms
        if n_t == 0 or n_atoms_tot == 0:
            logger.warning("Cannot calculate SED: 0 frames or 0 atoms.")
            return np.array([],dtype=np.complex64).reshape(0,0,3), np.array([],dtype=np.float32), True

        mean_pos_all = np.mean(self.traj.positions, axis=0, dtype=np.float32)
        freqs = np.fft.fftfreq(n_t, d=self.dt_ps) if n_t > 0 else np.array([],dtype=np.float32)

        # Determine atom groups for SED calculation
        atom_groups: List[np.ndarray] = []

        if basis_atom_types is not None:
            if basis_atom_indices is not None:
                logger.warning("Both basis_atom_types and basis_atom_indices provided. Using basis_atom_types.")
            
            # Ensure basis_atom_types is a list of lists for consistent group processing
            processed_basis_atom_types: List[List[int]] = []
            if isinstance(basis_atom_types, list) and len(basis_atom_types) > 0:
                if all(isinstance(item, list) for item in basis_atom_types):
                    processed_basis_atom_types = basis_atom_types # Already list of lists
                elif all(isinstance(item, int) for item in basis_atom_types):
                    if summation_mode == 'incoherent': # Each type is a group
                        processed_basis_atom_types = [[t] for t in basis_atom_types]
                    else: # All types form a single group for coherent sum
                        processed_basis_atom_types = [list(basis_atom_types)]
                else:
                    raise ValueError("basis_atom_types must be a list of ints or a list of lists of ints.")
            elif isinstance(basis_atom_types, int): # Single type treated as a single group
                 processed_basis_atom_types = [[basis_atom_types]]

            for type_group in processed_basis_atom_types:
                indices = np.where(np.isin(self.traj.types, type_group))[0]
                if indices.size > 0:
                    atom_groups.append(indices)
                else:
                    logger.warning(f"No atoms found for type group {type_group}. Skipping.")
        
        elif basis_atom_indices is not None:
            processed_basis_atom_indices: List[np.ndarray] = []
            if isinstance(basis_atom_indices, list):
                if len(basis_atom_indices) == 0:
                    pass # No specific indices, will use all atoms later if no groups formed
                elif all(isinstance(item, list) for item in basis_atom_indices): # List of lists of indices
                    for sublist in basis_atom_indices:
                        arr = np.asarray(sublist, dtype=int)
                        if arr.size > 0 : processed_basis_atom_indices.append(arr)
                elif all(isinstance(item, int) for item in basis_atom_indices): # Flat list of indices
                    arr = np.asarray(basis_atom_indices, dtype=int)
                    if arr.size > 0: processed_basis_atom_indices.append(arr) # Treated as a single group
                else:
                    raise ValueError("basis_atom_indices must be a list of ints or a list of lists of ints.")
            elif isinstance(basis_atom_indices, np.ndarray):
                if basis_atom_indices.ndim == 1 and basis_atom_indices.size > 0:
                     processed_basis_atom_indices.append(basis_atom_indices.astype(int))
                # Add handling for 2D ndarray if needed, treating rows as groups
                else:
                    logger.warning("Unsupported np.ndarray format for basis_atom_indices. Using all atoms if no other basis defined.")
            
            for grp_idx in processed_basis_atom_indices:
                if np.any(grp_idx >= n_atoms_tot) or np.any(grp_idx < 0):
                    raise ValueError("Atom indices in basis out of bounds.")
                if grp_idx.size > 0:
                    atom_groups.append(grp_idx)

        # If no groups defined by basis_atom_types or basis_atom_indices, use all atoms as a single group
        if not atom_groups:
            logger.debug(f"No specific basis provided or basis resulted in empty groups. Using all {n_atoms_tot} atoms as a single group.")
            atom_groups.append(np.arange(n_atoms_tot))
            if summation_mode == 'incoherent' and n_atoms_tot > 0:
                # If incoherent mode is on and we fell back to all atoms, warn it will be coherent sum of all atoms.
                logger.info("Using all atoms. Incoherent sum will effectively be a coherent sum of all atoms.")
        
        # Logic for coherent vs incoherent sum based on atom_groups
        if summation_mode == 'coherent' or len(atom_groups) <= 1:
            # Combine all groups into one for a single coherent calculation
            if len(atom_groups) > 1:
                logger.debug(f"Coherent mode with {len(atom_groups)} basis groups. Combining them for SED calculation.")
                final_group_indices = np.unique(np.concatenate(atom_groups)).astype(int)
            elif len(atom_groups) == 1:
                final_group_indices = atom_groups[0]
            else: # Should not happen if fallback to all atoms works
                final_group_indices = np.array([], dtype=int)

            if final_group_indices.size == 0:
                 logger.warning("Final atom group for SED calculation is empty. SED will be zero.")
                 return np.zeros((n_t, len(k_vectors_3d), 3), dtype=np.complex64), freqs, True

            logger.debug(f"Calculating SED coherently for {len(final_group_indices)} atoms.")
            sed_data = self._calculate_sed_for_group(k_vectors_3d, final_group_indices, mean_pos_all)
            is_complex_output = True
        else: # Incoherent sum over multiple groups
            logger.debug(f"Calculating SED incoherently by summing intensities from {len(atom_groups)} groups.")
            # Initialize accumulator for summed intensities (real, positive)
            summed_sed_intensity_data = np.zeros((n_t, len(k_vectors_3d), 3), dtype=np.float32)
            for i_grp, grp_indices in enumerate(atom_groups):
                logger.debug(f"  Calculating for group {i_grp+1}/{len(atom_groups)} with {len(grp_indices)} atoms.")
                sed_group_complex = self._calculate_sed_for_group(k_vectors_3d, grp_indices, mean_pos_all)
                summed_sed_intensity_data += np.abs(sed_group_complex)**2
            sed_data = summed_sed_intensity_data
            is_complex_output = False

        return sed_data.astype(np.complex64 if is_complex_output else np.float32), freqs, is_complex_output

    def calculate_chiral_phase(self, Z1: np.ndarray, Z2: np.ndarray, angle_range_opt: str = "C") -> np.ndarray:
        if Z1.shape != Z2.shape: 
            raise ValueError("Z1 and Z2 shapes must match for chiral phase.")
        if Z1.size == 0: 
            return np.array([], dtype=np.float32).reshape(Z1.shape)

        if angle_range_opt == "C": 
            p1, p2 = np.angle(Z1), np.angle(Z2)
            delta_p = p1 - p2
            delta_p = (delta_p + np.pi) % (2*np.pi) - np.pi # Wrap to [-pi, pi]
            delta_p[delta_p > (np.pi/2)] = np.pi - delta_p[delta_p > (np.pi/2)]   # Fold Q2
            delta_p[delta_p < (-np.pi/2)] = -np.pi - delta_p[delta_p < (-np.pi/2)] # Fold Q3
            return delta_p.astype(np.float32)
        else: 
            nw, nk = Z1.shape
            out_phase = np.zeros((nw,nk), dtype=np.float32)
            for i in range(nw):
                for j in range(nk):
                    v1r,v1i = Z1[i,j].real, Z1[i,j].imag
                    v2r,v2i = Z2[i,j].real, Z2[i,j].imag
                    m1sq,m2sq = v1r**2+v1i**2, v2r**2+v2i**2
                    if m1sq<1e-18 or m2sq<1e-18: 
                        angle=0.0
                    else:
                        m1,m2 = np.sqrt(m1sq), np.sqrt(m2sq)
                        if angle_range_opt == "A": 
                            angle = np.arccos(np.clip((v1r*v2r+v1i*v2i)/(m1*m2), -1.0, 1.0))
                        elif angle_range_opt == "B": 
                            angle = np.arcsin(np.clip((v1r*v2i-v1i*v2r)/(m1*m2), -1.0, 1.0))
                        else: 
                            logger.warning(f"Unknown angle_range_opt '{angle_range_opt}'. Angle=0.")
                            angle=0.0
                    out_phase[i,j] = angle
            return out_phase

    def ised(self, k_dir_spec: Union[str, int, float, List[float], np.ndarray, Dict[str,float]],
             k_target: float, w_target: float, char_len_k_path: float,
             nk_on_path: int = 100, bz_cov_ised: float = 1.0,
             basis_atom_idx_ised: Optional[List[int]] = None, 
             basis_atom_types_ised: Optional[List[int]] = None,
             rescale_factor: Union[str, float] = 1.0, n_recon_frames: int = 100,
             dump_filepath: str = "iSED_reconstruction.dump",
             plot_dir_ised: Optional[Path] = None, plot_max_freq: Optional[float] = None,
             plot_theme: str = 'light'
             ) -> None:
        logger.info("Starting iSED reconstruction.")
        avg_pos = np.mean(self.traj.positions, axis=0, dtype=np.float32)
        sys_atom_types = self.traj.types.astype(int)
        n_atoms_total = self.traj.n_atoms
        k_dir_unit = parse_direction(k_dir_spec)

        recon_atom_groups: List[np.ndarray] = []
        if basis_atom_idx_ised and len(basis_atom_idx_ised) > 0:
            if isinstance(basis_atom_idx_ised[0], list): 
                logger.info(f"iSED using specified atom index groups: {len(basis_atom_idx_ised)} groups.")
                for grp_idx in basis_atom_idx_ised:
                    grp_arr = np.asarray(grp_idx, dtype=int)
                    if np.any(grp_arr >= n_atoms_total) or np.any(grp_arr < 0): 
                        raise ValueError(f"Atom indices in group {grp_idx} out of bounds.")
                    if grp_arr.size > 0: 
                        recon_atom_groups.append(grp_arr)
            else: 
                logger.info(f"iSED using single atom index group ({len(basis_atom_idx_ised)} atoms).")
                grp_arr = np.asarray(basis_atom_idx_ised, dtype=int)
                if np.any(grp_arr >= n_atoms_total) or np.any(grp_arr < 0): 
                    raise ValueError("Atom indices out of bounds.")
                if grp_arr.size > 0: 
                    recon_atom_groups.append(grp_arr)
            if basis_atom_types_ised and len(basis_atom_types_ised) > 0:
                logger.warning("iSED: atom_indices and atom_types provided. Using atom_indices.")
        elif basis_atom_types_ised and len(basis_atom_types_ised) > 0:
            if isinstance(basis_atom_types_ised[0], list): 
                logger.info(f"iSED using specified atom type groups: {len(basis_atom_types_ised)} groups.")
                for type_grp in basis_atom_types_ised:
                    mask = np.isin(sys_atom_types, type_grp)
                    grp_idx = np.where(mask)[0]
                    if grp_idx.size > 0: 
                        recon_atom_groups.append(grp_idx)
                    else: 
                        logger.warning(f"No atoms for type group {type_grp} in iSED.")
            else: 
                logger.info(f"iSED using each atom type as a group for types: {basis_atom_types_ised}.")
                for atom_type_val in basis_atom_types_ised:
                    mask = np.isin(sys_atom_types, [atom_type_val])
                    grp_idx = np.where(mask)[0]
                    if grp_idx.size > 0: 
                        recon_atom_groups.append(grp_idx)
                    else: 
                        logger.warning(f"No atoms for type {atom_type_val} in iSED.")
        else:
            logger.info("iSED using all atoms as a single group.")
            recon_atom_groups.append(np.arange(n_atoms_total))

        if not recon_atom_groups: 
            logger.error("iSED: No atom groups for reconstruction. Aborting.")
            return

        logger.debug(f"iSED k-path: dir={k_dir_spec}, L_char={char_len_k_path}, nk={nk_on_path}, bz_cov={bz_cov_ised}")
        k_mags_ised, k_vecs_ised = self.get_k_path(direction_spec=k_dir_unit, bz_coverage=bz_cov_ised,
                                                 n_k=nk_on_path, lat_param=char_len_k_path)
        
        wiggles = np.zeros((n_recon_frames, n_atoms_total, 4), dtype=np.float32) # x,y,z,type
        time_p = np.linspace(0, 2*np.pi, n_recon_frames, endpoint=False)
        pos_proj_k_dir = np.dot(avg_pos, k_dir_unit)

        k_match_idx = np.argmin(np.abs(k_mags_ised - k_target))
        k_actual = k_mags_ised[k_match_idx]
        logger.info(f"iSED: Target k={k_target:.4f} -> Matched k={k_actual:.4f} (2π/Å, idx {k_match_idx})")

        recon_done, max_wiggle_amp_all = False, 0.0
        std_dev_sum, n_atoms_recon_sum = 0.0, 0
        ised_input_intensity_plot, ised_input_freqs_plot = None, None

        for i_grp, grp_atom_idx in enumerate(recon_atom_groups):
            if grp_atom_idx.size == 0: 
                logger.debug(f"Skipping empty iSED group {i_grp+1}.")
                continue
            
            grp_types_str = np.unique(sys_atom_types[grp_atom_idx])
            logger.info(f"iSED Group {i_grp+1}/{len(recon_atom_groups)}: {len(grp_atom_idx)} atoms (types: {grp_types_str}).")
            logger.debug(f"  iSED Group {i_grp+1}: Calculating SED for {len(grp_atom_idx)} atoms.")
            sed_group_data, freqs_group, is_complex = self.calculate(k_mags_ised, k_vecs_ised, basis_atom_indices=grp_atom_idx)
            
            if ised_input_freqs_plot is None: 
                ised_input_freqs_plot = freqs_group
            elif not np.array_equal(ised_input_freqs_plot, freqs_group): 
                logger.warning("iSED group freq arrays differ. Plotting may be inconsistent.")

            grp_intensity = np.sum(np.abs(sed_group_data)**2, axis=-1)
            if ised_input_intensity_plot is None: 
                ised_input_intensity_plot = grp_intensity.copy()
            else:
                if ised_input_intensity_plot.shape == grp_intensity.shape: 
                    ised_input_intensity_plot += grp_intensity
                else: 
                    logger.warning(f"iSED group intensity shape mismatch (group {i_grp+1}). Skipping accumulation.")

            w_match_idx = np.argmin(np.abs(freqs_group - w_target))
            w_actual = freqs_group[w_match_idx]
            logger.info(f"  iSED Group {i_grp+1}: Target ω={w_target:.3f} -> Matched ω={w_actual:.3f} (THz, idx {w_match_idx})")

            if grp_atom_idx.size > 0:
                logger.debug(f"  DEBUG Group {i_grp+1}: Spatial phase for k={k_actual:.4f} (rad/Å):")
                for atom_sys_idx in grp_atom_idx[:min(5, len(grp_atom_idx))]: 
                    r_proj = pos_proj_k_dir[atom_sys_idx]
                    spatial_p = k_actual * r_proj
                    logger.debug(f"    Atom SysIdx={atom_sys_idx}, Type={sys_atom_types[atom_sys_idx]}, r_proj={r_proj:.4f}Å, k*r_proj={spatial_p:.4f}rad")

            for pol_ax in range(3):
                sed_pol_data = sed_group_data[:, :, pol_ax]
                complex_amp_grp = sed_pol_data[w_match_idx, k_match_idx]
                proj_pos_grp = pos_proj_k_dir[grp_atom_idx]
                recon_motion_comp = np.real(complex_amp_grp * np.exp(1j * time_p[:,None] - 1j * k_actual * proj_pos_grp[None,:]))
                wiggles[:, grp_atom_idx, pol_ax] += recon_motion_comp
            
            recon_done = True
            if isinstance(rescale_factor, str) and rescale_factor.lower() == "auto":
                max_amp_grp = np.amax(np.abs(wiggles[:, grp_atom_idx, :3])) if grp_atom_idx.size > 0 else 0.0
                max_wiggle_amp_all = max(max_wiggle_amp_all, max_amp_grp)
                if grp_atom_idx.size > 0:
                    orig_disp_grp = self.traj.positions[:, grp_atom_idx,:] - avg_pos[None, grp_atom_idx,:]
                    std_dev_sum += np.std(orig_disp_grp) * len(grp_atom_idx)
                    n_atoms_recon_sum += len(grp_atom_idx)

        if not recon_done: 
            logger.error("iSED: No reconstruction performed (empty atom groups?).")
            return

        wiggles[0,:,3] = sys_atom_types # Store types in 4th component of 1st frame
        all_recon_idx = np.unique(np.concatenate(recon_atom_groups)) if recon_atom_groups and any(g.size > 0 for g in recon_atom_groups) else np.array([])
        
        if all_recon_idx.size > 0:
            if isinstance(rescale_factor, str) and rescale_factor.lower() == "auto":
                if max_wiggle_amp_all > 1e-9:
                    wiggles[:,all_recon_idx,:3] /= max_wiggle_amp_all 
                    avg_std_dev_disp = std_dev_sum / n_atoms_recon_sum if n_atoms_recon_sum > 0 else 0.0
                    if avg_std_dev_disp > 1e-9: 
                        wiggles[:,all_recon_idx,:3] *= avg_std_dev_disp
                    logger.info(f"iSED: Auto-rescaled. Max amp: {max_wiggle_amp_all:.3e}, Avg StdDev scale: {avg_std_dev_disp:.3e}")
                else: 
                    logger.warning("iSED: Max wiggle amp near zero. Auto-rescaling ineffective.")
            elif isinstance(rescale_factor, (int, float)):
                wiggles[:,all_recon_idx,:3] *= rescale_factor
                logger.info(f"iSED: Rescaled wiggles by factor {rescale_factor}.")
        else: 
            logger.warning("iSED: No atoms reconstructed, skipping rescaling.")

        final_pos_dump = avg_pos[None,:,:] + wiggles[:,:,:3]
        atom_types_dump = wiggles[0,:,3].astype(int) # Ensure types are integer for dump
        
        # Pass the full box_matrix for correct triclinic box representation
        out_to_qdump(dump_filepath, final_pos_dump, atom_types_dump, self.traj.box_matrix)
        logger.info(f"iSED reconstruction saved: {dump_filepath}")

        if plot_dir_ised and ised_input_intensity_plot is not None and ised_input_freqs_plot is not None:
            logger.info("Plotting iSED input spectrum (incoherently summed groups).")
            ised_mock_sed_plot = np.zeros((*ised_input_intensity_plot.shape, 3), dtype=np.complex64)
            ised_mock_sed_plot[:,:,0] = np.sqrt(ised_input_intensity_plot + 1e-20) # Store in first pol for SED object structure
            ised_plot_obj = SED(sed=ised_mock_sed_plot, freqs=ised_input_freqs_plot,
                                k_points=k_mags_ised, k_vectors=k_vecs_ised,
                                is_complex=True) # Mock SED is technically complex here, intensity handled by plotter
            
            # --- Filename Generation --- 
            k_dir_str = ""
            if isinstance(k_dir_spec, str):
                k_dir_str = k_dir_spec.replace(" ","_").replace("/", "-")
            elif isinstance(k_dir_spec, (list, tuple, np.ndarray)):
                arr = np.asarray(k_dir_spec)
                k_dir_str = f"({','.join([f'{x:.2f}' for x in arr])})"
            elif isinstance(k_dir_spec, dict):
                k_dir_str = f"(h{k_dir_spec.get('h',0)}_k{k_dir_spec.get('k',0)}_l{k_dir_spec.get('l',0)})"
            else:
                k_dir_str = str(k_dir_spec)
            k_dir_str = k_dir_str.replace("[", "").replace("]", "").replace("(", "").replace(")", "") # Clean common brackets

            k_target_str = f"{k_target:.2f}".replace('.','p')
            w_target_str = f"{w_target:.2f}".replace('.','p')
            
            # New filename pattern:
            ised_plot_fname = plot_dir_ised / f"iSED_{k_dir_str}_{k_target_str}_{w_target_str}.png"
            # Old filename pattern for reference (was more complex):
            # ised_plot_fname = plot_dir_ised / f"ised_input_sed_{ised_k_dir_label}_k{k_str}_w{w_str}_inc_sum.png"
            
            w_match_plot_idx = np.argmin(np.abs(ised_input_freqs_plot - w_target))
            w_actual_plot = ised_input_freqs_plot[w_match_plot_idx]
            hl_info = {'k_point_target':k_actual, 'freq_point_target':w_actual_plot}
            
            max_freq_ised_plot = plot_max_freq
            if max_freq_ised_plot is None and ised_input_freqs_plot.size > 0:
                 max_freq_ised_plot = np.max(ised_input_freqs_plot)
            
            plot_args_ised = {
                'title': f"Summed iSED Input Spectrum (k≈{k_actual:.3f}, ω≈{w_actual_plot:.3f})",
                'direction_label': k_dir_str, # Use the formatted k_dir_str for label consistency
                'highlight_region': hl_info,
                'max_freq': max_freq_ised_plot,
                'log_intensity': True,  # Enable log scaling for intensity
                'theme': plot_theme  # Pass theme to SEDPlotter
            }
            SEDPlotter(ised_plot_obj, '2d_intensity', str(ised_plot_fname), **plot_args_ised).generate_plot()
            logger.info(f"iSED input spectrum plot saved: {ised_plot_fname.name}")
        elif plot_dir_ised:
            logger.warning("iSED plot requested, but no combined SED data available.")


