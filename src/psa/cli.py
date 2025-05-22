import argparse
import yaml
from pathlib import Path
import numpy as np
import logging
from typing import List

# from psa.core.trajectory import Trajectory # This import seems unused in the main() context now
from psa.io.loader import TrajectoryLoader # Corrected path
from psa.core.sed_calculator import SEDCalculator
from psa.core.sed import SED
from psa.visualization.plotter import SEDPlotter
# from psa.utils.config_loader import update_dict_recursively # Old import
from psa.utils.helpers import update_dict_recursively # Corrected path

logger = logging.getLogger(__name__)

# Basic logging configuration (moved from original PSA.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(description='Phonon Spectral Analysis Tool.')
    parser.add_argument('--trajectory', type=str, required=True, help='Path to MD trajectory file.')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file.')
    parser.add_argument('--output-dir', type=str, default='psa_output', help='Directory for results.')
    parser.add_argument('--chiral', action='store_true', help='Enable chiral SED (overrides config).')
    parser.add_argument('--dt', type=float, help="Override MD timestep from config (ps).")
    parser.add_argument('--nk', type=int, help="Override n_kpoints for SED from config.")
    parser.add_argument('--recalculate-sed', action='store_true', help="Force recalculation of SED data.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    default_config = {
        'general': {'trajectory_file_format':'auto', 'use_displacements':False, 'save_npy_trajectory':True, 'save_npy_sed_data':True, 'chiral_mode_enabled':False},
        'md_system': {'dt':0.001, 'nx':1, 'ny':1, 'nz':1, 'lattice_parameter':None},
        'sed_calculation': {'directions':[[1,0,0]], 'n_kpoints':100, 'bz_coverage':1.0, 'polarization_indices_chiral':[0,1], 'basis':{'atom_indices':None, 'atom_types':None}},
        'plotting': {'max_freq_2d':None, 'highlight_2d_intensity':{'k_min':None,'k_max':None,'w_min':None,'w_max':None}, 'enable_3d_dispersion_plot':True, '3d_plot_settings':{'intensity_log_scale':True, 'intensity_thresh_rel':0.05}},
        'ised': {'apply':False, 'k_path':{'direction':'x', 'characteristic_length':None, 'n_points':50, 'bz_coverage':None}, 'target_point':{'k_value':6.283, 'w_value_thz':10.0}, 'basis':{'atom_indices':None, 'atom_types':None}, 'reconstruction':{'rescaling_factor':'auto', 'num_animation_timesteps':100, 'output_dump_filename':'ised_motion.dump'}}
    }
    config = default_config.copy()

    if args.config:
        try:
            with open(args.config, 'r') as f: user_cfg = yaml.safe_load(f)
            if user_cfg: config = update_dict_recursively(config, user_cfg)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e: logger.error(f"Error loading config {args.config}: {e}. Using defaults.")

    if args.dt is not None: config['md_system']['dt'] = args.dt
    if args.nk is not None: config['sed_calculation']['n_kpoints'] = args.nk
    if args.chiral: config['general']['chiral_mode_enabled'] = True 

    gen_cfg, md_cfg, sed_cfg, plot_cfg, ised_cfg = \
        config['general'], config['md_system'], config['sed_calculation'], config['plotting'], config['ised']

    if md_cfg['dt'] <= 0: logger.error("Timestep 'dt' must be positive."); raise SystemExit(1)
    
    try:
        logger.info(f"Loading trajectory: {args.trajectory} (dt={md_cfg['dt']:.4f} ps)")
        traj_load = TrajectoryLoader(args.trajectory, dt=md_cfg['dt'], file_format=gen_cfg['trajectory_file_format'])
        traj_data = traj_load.load()
        if gen_cfg['save_npy_trajectory']: traj_load.save_trajectory_npy(traj_data)

        sed_calc = SEDCalculator(traj=traj_data, nx=md_cfg['nx'], ny=md_cfg['ny'], nz=md_cfg['nz'],
                                 dt_ps=md_cfg['dt'], use_displacements=gen_cfg['use_displacements'])
        
        eff_lat_param = md_cfg.get('lattice_parameter')
        if eff_lat_param is None or eff_lat_param <= 1e-6:
            norm_a1_val = np.linalg.norm(sed_calc.a1)
            if norm_a1_val > 1e-6: eff_lat_param = norm_a1_val; logger.info(f"Using |a1| ({eff_lat_param:.3f} Å) as effective lattice parameter.")
            else: raise ValueError("Cannot determine valid effective_lattice_parameter. Specify in config or check box/nx,ny,nz.")
        md_cfg['lattice_parameter'] = eff_lat_param 

        main_sed_basis_idx = None; main_basis_cfg = sed_cfg['basis']
        main_idx_spec, main_types_spec = main_basis_cfg.get('atom_indices'), main_basis_cfg.get('atom_types')
        if main_idx_spec and len(main_idx_spec) > 0:
            main_sed_basis_idx = np.asarray(main_idx_spec, dtype=int)
            if main_types_spec and len(main_types_spec) > 0: logger.warning("Main SED: atom_indices and atom_types specified; using atom_indices.")
        elif main_types_spec and len(main_types_spec) > 0:
            type_m = np.isin(traj_data.types, main_types_spec); main_sed_basis_idx = np.where(type_m)[0]
            if not main_sed_basis_idx.size: logger.warning(f"Main SED: No atoms for types {main_types_spec}. Using all."); main_sed_basis_idx = None
        if main_sed_basis_idx is not None and (np.any(main_sed_basis_idx >= traj_data.n_atoms) or np.any(main_sed_basis_idx < 0)):
            raise ValueError("Main SED basis indices out of bounds.")

        global_max_i = None; dirs_list = sed_cfg['directions']
        if len(dirs_list) > 1 and not gen_cfg['chiral_mode_enabled']:
            logger.info("Calculating global max intensity for plot normalization...")
            max_i_vals = []
            for dir_s in dirs_list:
                k_mags_norm, k_vecs_norm = sed_calc.get_k_path(dir_s, sed_cfg['bz_coverage'], sed_cfg['n_kpoints'], eff_lat_param)
                sed_obj_norm = sed_calc.calculate(k_points_mags=k_mags_norm, 
                                                  k_vectors_3d=k_vecs_norm, 
                                                  basis_atom_indices=main_sed_basis_idx,
                                                  k_grid_shape=None)
                sed_complex_norm = sed_obj_norm.sed
                curr_i = np.sum(np.abs(sed_complex_norm)**2, axis=-1)
                if curr_i.size > 0: max_i_vals.append(np.max(curr_i))
            if max_i_vals: global_max_i = np.max(max_i_vals)
            logger.info(f"Global max intensity: {global_max_i:.4e}" if global_max_i else "Not determined.")

        all_sed_results = []
        for i_d, dir_val_spec in enumerate(dirs_list, 1):
            if isinstance(dir_val_spec,(int,float)): d_lbl=f"{dir_val_spec:.1f}deg"
            elif isinstance(dir_val_spec,str): d_lbl=dir_val_spec.replace(" ","_").replace("/","-")
            elif isinstance(dir_val_spec,(list,tuple,np.ndarray)): arr_d=np.asarray(dir_val_spec); d_lbl=f"{arr_d.item():.1f}deg" if arr_d.size==1 else '_'.join([f"{x:.2f}" for x in arr_d])
            elif isinstance(dir_val_spec,dict): d_lbl=f"h{dir_val_spec.get('h',0)}_k{dir_val_spec.get('k',0)}_l{dir_val_spec.get('l',0)}"
            else: d_lbl = f"dir{i_d}"
            logger.info(f"Processing direction {i_d}/{len(dirs_list)}: {d_lbl}")

            sed_res = None; sed_sfx = "chiral" if gen_cfg['chiral_mode_enabled'] else "regular"; basis_sfx = ""
            if main_sed_basis_idx is not None:
                if main_idx_spec and len(main_idx_spec) > 0: basis_sfx = "_idxbasis"
                elif main_types_spec and len(main_types_spec) > 0: basis_sfx = f"_typebasis{'_'.join(map(str,main_types_spec))}"
            sed_savefile_base = out_dir / f"sed_data_{sed_sfx}_{d_lbl}{basis_sfx}"

            if gen_cfg['save_npy_sed_data'] and not args.recalculate_sed:
                try: sed_res = SED.load(sed_savefile_base); logger.info(f"Loaded SED data for {d_lbl}.")
                except FileNotFoundError: logger.info(f"No pre-calculated SED for {d_lbl}. Will calculate.")
                except Exception as e: logger.warning(f"Failed to load SED for {d_lbl}: {e}. Recalculating.")
            
            needs_recalc_for_phase = gen_cfg['chiral_mode_enabled'] and (sed_res is None or (sed_res.phase is None and not sed_savefile_base.with_suffix('.phase.npy').exists()))
            if sed_res is None or needs_recalc_for_phase:
                if needs_recalc_for_phase and sed_res: logger.info(f"Recalculating SED for {d_lbl} (phase data needed).")
                k_m, k_v = sed_calc.get_k_path(dir_val_spec, sed_cfg['bz_coverage'], sed_cfg['n_kpoints'], eff_lat_param)
                sed_obj_calc = sed_calc.calculate(k_points_mags=k_m, 
                                                  k_vectors_3d=k_v, 
                                                  basis_atom_indices=main_sed_basis_idx,
                                                  k_grid_shape=None)
                sed_complex = sed_obj_calc.sed
                freqs_arr = sed_obj_calc.freqs

                phase_arr = None
                if gen_cfg['chiral_mode_enabled']:
                    pol_idx_chiral = sed_cfg['polarization_indices_chiral']
                    if len(pol_idx_chiral) >= 2 and sed_complex.shape[-1] > max(pol_idx_chiral):
                        phase_arr = sed_calc.calculate_chiral_phase(sed_complex[:,:,pol_idx_chiral[0]], sed_complex[:,:,pol_idx_chiral[1]])
                    else: logger.error(f"Chiral mode error for {d_lbl}: Insufficient polarizations or invalid indices {pol_idx_chiral}.")
                sed_res = SED(sed=sed_obj_calc.sed,
                              freqs=sed_obj_calc.freqs,
                              k_points=sed_obj_calc.k_points,
                              k_vectors=sed_obj_calc.k_vectors,
                              phase=phase_arr,
                              is_complex=sed_obj_calc.is_complex,
                              k_grid_shape=sed_obj_calc.k_grid_shape,
                              dt_ps=sed_obj_calc.dt_ps,
                              trajectory_metadata=sed_obj_calc.trajectory_metadata)
                if gen_cfg['save_npy_sed_data']: sed_res.save(sed_savefile_base)
            
            all_sed_results.append(sed_res)
            plot_args = {'direction_label': d_lbl, 'max_freq': plot_cfg['max_freq_2d']}
            if gen_cfg['chiral_mode_enabled']:
                if sed_res.phase is not None: SEDPlotter(sed_res, '2d_phase', str(out_dir/f"sed_phase_2D_{d_lbl}{basis_sfx}.png"), **plot_args).generate_plot()
                else: logger.info(f"Skipping 2D phase plot for {d_lbl} (no phase data).")
            else:
                plot_args['global_max_intensity_val'] = global_max_i
                hl_spec = plot_cfg['highlight_2d_intensity']
                if all(v is not None for v in [hl_spec['k_min'],hl_spec['k_max'],hl_spec['w_min'],hl_spec['w_max']]):
                    try: plot_args['highlight_region'] = {'k_range':(float(hl_spec['k_min']),float(hl_spec['k_max'])), 'freq_range':(float(hl_spec['w_min']),float(hl_spec['w_max']))}
                    except ValueError: logger.warning("Invalid highlight region parameters for 2D intensity plot.")
                SEDPlotter(sed_res, '2d_intensity', str(out_dir/f"sed_intensity_2D_{d_lbl}{basis_sfx}.png"), **plot_args).generate_plot()

        if plot_cfg['enable_3d_dispersion_plot'] and all_sed_results:
            logger.info("Generating 3D dispersion plots...")
            basis_sfx_3d = ""
            if main_sed_basis_idx is not None:
                if main_idx_spec and len(main_idx_spec) > 0 : basis_sfx_3d = "_idxbasis"
                elif main_types_spec and len(main_types_spec) > 0 : basis_sfx_3d = f"_typebasis{'_'.join(map(str,main_types_spec))}"
            
            plot_3d_cfg = plot_cfg['3d_plot_settings']; plot_3d_args = {'title': "3D SED Dispersion"} 
            if gen_cfg['chiral_mode_enabled']:
                seds_w_phase = [s for s in all_sed_results if s.phase is not None]
                if seds_w_phase: plot_3d_args['title'] += " (Phase)"; SEDPlotter(seds_w_phase, '3d_phase', str(out_dir/f"disp_3D_sed_phase{basis_sfx_3d}.png"), **plot_3d_args).generate_plot()
                else: logger.warning("No SED objects with phase data for 3D phase plot.")
            else:
                plot_3d_args['title'] += " (Intensity)"
                plot_3d_args['intensity_log_scale'] = plot_3d_cfg['intensity_log_scale']
                plot_3d_args['intensity_thresh_rel_gather'] = plot_3d_cfg['intensity_thresh_rel']
                SEDPlotter(all_sed_results, '3d_intensity', str(out_dir/f"disp_3D_sed_intensity{basis_sfx_3d}.png"), **plot_3d_args).generate_plot()
        
        if ised_cfg['apply']:
            logger.info("Performing iSED reconstruction...")
            ised_kpath_cfg, ised_tgt_cfg, ised_basis_cfg, ised_recon_cfg = \
                ised_cfg['k_path'], ised_cfg['target_point'], ised_cfg['basis'], ised_cfg['reconstruction']
            
            char_len_k_ised = ised_kpath_cfg['characteristic_length'] or md_cfg['lattice_parameter']
            bz_cov_k_ised = ised_kpath_cfg['bz_coverage'] or sed_cfg['bz_coverage']
            max_freq_for_ised_plot = plot_cfg.get('max_freq_2d') 

            sed_calc.ised(
                k_dir_spec=ised_kpath_cfg['direction'], k_target=float(ised_tgt_cfg['k_value']),
                w_target=float(ised_tgt_cfg['w_value_thz']), char_len_k_path=float(char_len_k_ised),
                nk_on_path=int(ised_kpath_cfg['n_points']), bz_cov_ised=float(bz_cov_k_ised),
                basis_atom_idx_ised=ised_basis_cfg.get('atom_indices'), basis_atom_types_ised=ised_basis_cfg.get('atom_types'),
                rescale_factor=ised_recon_cfg['rescaling_factor'], n_recon_frames=int(ised_recon_cfg['num_animation_timesteps']),
                dump_filepath=str(out_dir / ised_recon_cfg['output_dump_filename']),
                plot_dir_ised=out_dir, plot_max_freq=max_freq_for_ised_plot
            )
        logger.info("PSA processing completed.")

    except FileNotFoundError as e: logger.error(f"File Error: {e}"); raise SystemExit(1)
    except ValueError as e: logger.error(f"Value Error: {e}"); raise SystemExit(1)
    except Exception as e: logger.error(f"Unexpected error: {e}", exc_info=True); raise SystemExit(1)

if __name__ == "__main__":
    main() 