"""
Visualization module for SED data.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple
import logging

from ..core.sed import SED

logger = logging.getLogger(__name__)

class SEDPlotter:
    def __init__(self, sed_obj: SED, plot_type: str, output_path: str, **kwargs):
        """
        Initialize SEDPlotter with SED data and plotting parameters.
        
        Args:
            sed_obj: SED object containing data to plot
            plot_type: Type of plot to generate ('2d_intensity', '1d_slice', 'frequency_slice', etc.)
            output_path: Path to save the plot
            **kwargs: Additional plotting parameters
        """
        self.sed = sed_obj
        self.plot_type = plot_type
        self.output_path = Path(output_path)
        self.plot_params = kwargs
        
        # Default parameters
        self.default_params = {
            'title': 'SED Spectrum',
            'xlabel': r'k ($2\pi/\AA$)',
            'ylabel': 'Frequency (THz)',
            'cmap': 'inferno',
            'figsize': (10, 8),
            'dpi': 300,
            'max_freq': None,
            'target_frequency': 1.0, # Default target frequency for frequency_slice
            'k_index': None, # For 1d_slice
            'freq_index': None, # For 1d_slice
            'highlight_region': None,
            'direction_label': '',
            'show_colorbar': True,
            'colorbar_label': 'Intensity (arb. units)',
            'grid': True,
            'tight_layout': True,
            'log_intensity': False,
            'vmin_percentile': 0.0,
            'vmax_percentile': 100.0,
            'theme': 'light'  # Added theme parameter, default to light
        }
        
        # Update with user parameters
        self.plot_params = {**self.default_params, **kwargs}
        
    def generate_plot(self) -> None:
        fig, ax = None, None
        
        # Apply theme settings to matplotlib's rcParams
        theme = self.plot_params.get('theme', 'light')
        if theme == 'dark':
            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = 'black'
            plt.rcParams['xtick.color'] = 'white'
            plt.rcParams['ytick.color'] = 'white'
            plt.rcParams['text.color'] = 'white'
            plt.rcParams['axes.labelcolor'] = 'white'
        else:
            plt.style.use('default')
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['xtick.color'] = 'black'
            plt.rcParams['ytick.color'] = 'black'
            plt.rcParams['text.color'] = 'black'
            plt.rcParams['axes.labelcolor'] = 'black'

        plot_method_name = f"_plot_{self.plot_type}"

        if hasattr(self, plot_method_name) and callable(getattr(self, plot_method_name)):
            plot_method = getattr(self, plot_method_name)
            if self.plot_type.startswith("2d_"):
                # Assuming 2D plots take no extra args for now
                fig, ax = plot_method()
            elif self.plot_type.startswith("3d_"):
                # Placeholder for 3D if it also returns fig, ax
                logger.warning(f"Plot type {self.plot_type} might require specific argument handling in generate_plot.")
                # Fallback or raise error if not handled
                raise ValueError(f"Plot type {self.plot_type} not fully supported in generate_plot yet.")
            else:
                # If a plot_method exists but isn't explicitly handled (e.g. not "2d_" or "3d_")
                # fig and ax will remain None, which is handled by the 'if fig:' check later.
                # Or, we could attempt to call it if it has a standard signature:
                # try:
                #     fig, ax = plot_method()
                # except Exception as e:
                #     logger.error(f"Error calling unhandled plot method {plot_method_name}: {e}")
                # For now, let fig, ax remain None if not 2d/3d
                pass
        else:
            # This is where the original SyntaxError was reported.
            # This 'else' corresponds to 'if hasattr...'
            raise ValueError(f"Unknown or non-callable plot type / method: {self.plot_type} (method: {plot_method_name})")
            
        if fig: # Proceed only if a figure was generated
            if self.plot_params.get('tight_layout', True): # Use get for safety
                fig.tight_layout()
            
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_path, dpi=self.plot_params.get('dpi', 300), bbox_inches='tight')
            plt.close(fig) # Close the figure to free memory
            logger.info(f"Plot saved to: {self.output_path}")
        else:
            logger.warning(f"Plot generation for {self.plot_type} did not return a figure. Output file {self.output_path} not created.")
        
    def _plot_2d_intensity(self) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Generate 2D intensity plot of SED data."""
        fig, ax = plt.subplots(figsize=self.plot_params['figsize'], dpi=self.plot_params.get('dpi', 300))
        self._setup_ax_style(fig, ax)

        # Calculate intensity
        if self.sed.is_complex:
            intensity_raw = np.sum(np.abs(self.sed.sed)**2, axis=-1)
        else:
            # If not complex, sed data is already summed intensities per polarization
            intensity_raw = np.sum(self.sed.sed, axis=-1) 
        
        # --- Frequency Masking (Positive and up to max_freq) ---
        positive_freq_mask = self.sed.freqs >= 0
        plot_freqs = self.sed.freqs[positive_freq_mask]
        intensity_at_pos_freqs = intensity_raw[positive_freq_mask]

        if self.plot_params['max_freq'] is not None:
            freq_upper_bound_mask = plot_freqs <= self.plot_params['max_freq']
            plot_freqs = plot_freqs[freq_upper_bound_mask]
            intensity_at_pos_freqs = intensity_at_pos_freqs[freq_upper_bound_mask]

        # --- K-points ---
        k_points_plot = self.sed.k_points
        if k_points_plot.ndim == 0:
             k_points_plot = np.array([k_points_plot])

        if plot_freqs.size == 0 or k_points_plot.size == 0:
            logger.warning(f"Not enough data for 2D intensity plot {self.output_path.name} (empty freqs or k_points after masking).")
            plt.close(fig)
            return None, None

        # --- Intensity Data Preparation (Log scaling, Normalization) ---
        intensity_to_plot = intensity_at_pos_freqs
        current_colorbar_label = self.plot_params['colorbar_label']

        if self.plot_params['log_intensity']:
            if np.any(intensity_to_plot > 1e-12): # Check if there's anything to log scale
                intensity_to_plot = np.log10(np.maximum(intensity_to_plot, 1e-12)) # Avoid log(0) or log(negative)
                current_colorbar_label = 'Log10(Intensity)'
            else:
                logger.warning("Log scaling requested for intensity, but all values are too small or zero. Using linear scale.")

        # Ensure intensity_to_plot has correct dimensions for meshgrid if it became 1D due to single k-point/freq
        if intensity_to_plot.ndim == 1 and k_points_plot.size > 1 and plot_freqs.size == 1: # Single freq, multiple k
            intensity_to_plot = intensity_to_plot[np.newaxis, :] # (1, Nk)
        elif intensity_to_plot.ndim == 1 and plot_freqs.size > 1 and k_points_plot.size == 1: # Single k, multiple freqs
            intensity_to_plot = intensity_to_plot[:, np.newaxis] # (Nf, 1)
        
        # If k_points_plot is scalar and plot_freqs is an array, intensity_to_plot should be (Nf, 1)
        # If plot_freqs is scalar and k_points_plot is an array, intensity_to_plot should be (1, Nk)
        # If both are scalar, it's (1,1)
        # The intensity_at_pos_freqs should already be (Nf_masked, Nk_total) before k-point selection for the plot

        # Correcting intensity_to_plot to match k_points_plot for the current plot.
        # Assuming sed.k_points are the full set and intensity_raw is (N_total_freqs, N_total_k_points)
        # intensity_at_pos_freqs is (N_positive_freqs_masked, N_total_k_points)
        # k_points_plot here refers to self.sed.k_points which are magnitudes for a 1D path.
        # So intensity_at_pos_freqs should already align with these k_points.

        K, F = np.meshgrid(k_points_plot, plot_freqs)
        
        if K.shape != intensity_to_plot.shape:
             # This logic might be too aggressive or based on wrong assumptions
             # Let's assume intensity_to_plot is (Nf_masked_final, Nk_final)
             # And k_points_plot is (Nk_final), plot_freqs is (Nf_masked_final)
             # The most common issue is if one of them is 1.
            logger.warning(f"Shape mismatch for pcolormesh, K={K.shape}, F={F.shape}, C={intensity_to_plot.shape}. Check data alignment. Attempting to plot anyway.")
            # If intensity_to_plot is (Nf, Nk) it should be fine.
            # If K,F are (Nf,Nk) and C is (Nf,Nk) pcolormesh works.
        
        # --- Vmin, Vmax Calculation (Percentile-based) ---
        valid_intensity_values = intensity_to_plot[~np.isnan(intensity_to_plot) & ~np.isinf(intensity_to_plot)]
        vmin, vmax = None, None
        if valid_intensity_values.size > 0:
            vmin = np.percentile(valid_intensity_values, self.plot_params['vmin_percentile'])
            vmax = np.percentile(valid_intensity_values, self.plot_params['vmax_percentile'])
            if vmin == vmax: # Handle case where all values are the same
                vmin = vmin - 0.1 if vmin != 0 else -0.1 # Avoid vmin=vmax=0
                vmax = vmax + 0.1 if vmax != 0 else 0.1
        
        # --- Plotting ---
        pcm = ax.pcolormesh(K, F, intensity_to_plot, 
                      cmap=self.plot_params['cmap'],
                           shading='gouraud', # Changed from 'auto'
                           vmin=vmin, vmax=vmax)
        
        # --- Labels and Title ---
        base_xlabel = self.plot_params['xlabel'] # This is r'k ($2\\pi/\\AA$)' by default
        if self.plot_params['direction_label']:
            # Ensure direction_label is a string before formatting
            direction_str = str(self.plot_params['direction_label'])
            xlabel_text = f"{direction_str} {base_xlabel}"
        else:
            xlabel_text = base_xlabel
        ax.set_xlabel(xlabel_text)
        ax.set_ylabel(self.plot_params['ylabel'])
        
        title_text = self.plot_params['title']
        # OriginalSDA adds direction_label to title if not used for xlabel. Here, direction_label makes xlabel specific.
        # If we want it in title too, that's an option:
        # if self.plot_params['direction_label'] and xlabel_text != self.plot_params['direction_label']:
        #     title_text += f"\\nDirection: {self.plot_params['direction_label']}"
        ax.set_title(title_text)

        # --- Y-axis Limit ---
        if plot_freqs.size > 0:
            # max_f_plot_val = np.max(plot_freqs) if np.max(plot_freqs) > 0 else 1.0
            # ax.set_ylim(0, max_f_plot_val)
            # If max_freq was used to mask, plot_freqs already respects it.
            # If max_freq wasn't specified, use max of plot_freqs.
            # If plot_freqs is empty after all, this won't run.
            max_y_limit = self.plot_params['max_freq'] if self.plot_params['max_freq'] is not None else np.max(plot_freqs)
            if max_y_limit > 0 : #Ensure positive ylim
                 ax.set_ylim(0, max_y_limit)
            else: # Fallback if max_y_limit is 0 or negative
                 ax.set_ylim(0,1)

        
        # Add highlight region if specified
        if self.plot_params['highlight_region']:
            hl = self.plot_params['highlight_region']
            if 'k_point_target' in hl and 'freq_point_target' in hl:
                ax.plot(hl['k_point_target'], hl['freq_point_target'], 
                        'g+', markersize=10, label='Target point') # Changed from 'r+' to 'g+'
                if self.plot_params.get('highlight_label', False): # OriginalSDA has show_target_label
                     ax.legend()
        
        # Add colorbar
        if self.plot_params['show_colorbar'] and hasattr(pcm, 'get_array') and pcm.get_array().size > 0:
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label(current_colorbar_label) # Use potentially updated label
            
        # Add grid
        if self.plot_params['grid']:
            ax.grid(True, alpha=0.3)
        
        return fig, ax
            
    def _plot_1d_slice(self) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """
        Generate 1D slice plot of SED data.
        Plots Intensity vs. Frequency for a given k-point index,
        or Intensity vs. K-point for a given frequency index.
        """
        fig, ax = plt.subplots(figsize=self.plot_params.get('figsize', (10, 6))) # Adjusted default figsize for 1D
        self._setup_ax_style(fig, ax)

        k_index = self.plot_params.get('k_index')
        freq_index = self.plot_params.get('freq_index')

        if k_index is None and freq_index is None:
            logger.error("Must specify either k_index or freq_index for 1D slice.")
            plt.close(fig)
            return None, None
            
        # Calculate intensity
        if self.sed.is_complex:
            intensity_data = np.sum(np.abs(self.sed.sed)**2, axis=-1)
        else:
            intensity_data = np.sum(self.sed.sed, axis=-1)
        
        plot_title = self.plot_params.get('title', '1D SED Slice')
        xlabel = ""
        ylabel = 'Intensity (arb. units)'
        
        if self.plot_params.get('log_intensity'):
            if np.any(intensity_data > 1e-12):
                intensity_data = np.log10(np.maximum(intensity_data, 1e-12))
                ylabel = 'Log10(Intensity)'
            else:
                logger.warning("Log scaling requested for intensity, but all values are too small or zero. Using linear scale.")

        if k_index is not None:
            if not (0 <= k_index < self.sed.k_points.shape[0]):
                logger.error(f"k_index {k_index} is out of bounds for k_points shape {self.sed.k_points.shape}")
                plt.close(fig)
                return None, None
            
            data_to_plot = intensity_data[:, k_index]
            x_values = self.sed.freqs
            
            xlabel = self.plot_params.get('ylabel', 'Frequency (THz)') # Note: using ylabel for x-axis here
            k_val_str = f"{self.sed.k_points[k_index]:.3f}"
            direction_label_str = str(self.plot_params.get('direction_label', ''))
            base_k_unit_label = self.plot_params.get('xlabel', r'k ($2\pi/\AA$)').split('(')[0].strip() # Get "k"
            
            label = f'{direction_label_str} {base_k_unit_label}={k_val_str} {self.plot_params.get("xlabel", r"($2\pi/\AA$)").split(" ", 1)[-1]}'

            plot_title = f"{plot_title}: Intensity vs Frequency"
            ax.plot(x_values, data_to_plot, label=label)

            # Apply max_freq if specified
            if self.plot_params.get('max_freq') is not None:
                ax.set_xlim(0, self.plot_params['max_freq']) # Assuming freqs start from 0 for this slice plot
            else:
                if x_values.size > 0 : ax.set_xlim(0, np.max(x_values))


        elif freq_index is not None:
            if not (0 <= freq_index < self.sed.freqs.shape[0]):
                logger.error(f"freq_index {freq_index} is out of bounds for freqs shape {self.sed.freqs.shape}")
                plt.close(fig)
                return None, None

            data_to_plot = intensity_data[freq_index, :]
            x_values = self.sed.k_points

            xlabel = self.plot_params.get('xlabel', r'k ($2\pi/\AA$)')
            direction_label_str = str(self.plot_params.get('direction_label', ''))
            if direction_label_str:
                xlabel = f"{direction_label_str} {xlabel}"

            label = f'ω = {self.sed.freqs[freq_index]:.3f} THz'
            plot_title = f"{plot_title}: Intensity vs K-points"
            ax.plot(x_values, data_to_plot, label=label)
        
        else: # Should not happen due to initial check, but as a safeguard
            plt.close(fig)
            return None, None

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(plot_title)
        
        if self.plot_params.get('grid', True):
            ax.grid(True, alpha=0.3)
        ax.legend()
        return fig, ax

    def _plot_frequency_slice(self) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """
        Generate 1D plot of SED intensity vs. k-points for a target frequency.
        """
        fig, ax = plt.subplots(figsize=self.plot_params.get('figsize', (10, 6)))
        self._setup_ax_style(fig, ax)

        target_freq = self.plot_params.get('target_frequency')
        if target_freq is None:
            logger.error("target_frequency must be specified for frequency_slice plot type.")
            plt.close(fig)
            return None, None

        # Find the closest frequency index
        if self.sed.freqs is None or self.sed.freqs.size == 0:
            logger.error("SED object has no frequency data.")
            plt.close(fig)
            return None, None
            
        freq_idx = np.argmin(np.abs(self.sed.freqs - target_freq))
        actual_freq = self.sed.freqs[freq_idx]

        # Calculate intensity for the selected frequency slice
        if self.sed.is_complex:
            # sed data shape: (n_freqs, n_kpoints, n_polarizations_or_groups)
            intensity_slice = np.sum(np.abs(self.sed.sed[freq_idx, :, :])**2, axis=-1)
        else:
            # sed data shape: (n_freqs, n_kpoints) or (n_freqs, n_kpoints, n_polarizations_already_summed_incoherently)
            # if last dim is polarizations, sum it. If it's already (n_freqs, n_kpoints), it's fine.
            if self.sed.sed.ndim == 3:
                 intensity_slice = np.sum(self.sed.sed[freq_idx, :, :], axis=-1)
            elif self.sed.sed.ndim == 2: # (n_freqs, n_kpoints)
                 intensity_slice = self.sed.sed[freq_idx, :]
            else:
                logger.error(f"Unsupported SED data format for frequency slice: ndim={self.sed.sed.ndim}")
                plt.close(fig)
                return None, None


        k_points_plot = self.sed.k_points
        if k_points_plot.ndim == 0: # Scalar k-point
            k_points_plot = np.array([k_points_plot])
        
        if k_points_plot.size == 0:
            logger.warning(f"No k-points found for frequency slice plot at {actual_freq:.2f} THz.")
            plt.close(fig)
            return None, None
            
        if intensity_slice.shape[0] != k_points_plot.shape[0]:
            logger.error(f"Shape mismatch: intensity_slice has shape {intensity_slice.shape} but k_points_plot has shape {k_points_plot.shape}")
            plt.close(fig)
            return None, None

        # Log scaling for y-axis (intensity)
        current_ylabel = 'Intensity (arb. units)'
        plot_data = intensity_slice
        if self.plot_params.get('log_intensity'):
            if np.any(plot_data > 1e-12):
                plot_data = np.log10(np.maximum(plot_data, 1e-12))
                current_ylabel = 'Log10(Intensity)'
            else:
                logger.warning("Log scaling requested for intensity, but all values are too small or zero. Using linear scale.")
        
        ax.plot(k_points_plot, plot_data)
        
        # Labels and Title
        base_xlabel = self.plot_params.get('xlabel', r'k ($2\pi/\AA$)')
        direction_label_str = str(self.plot_params.get('direction_label', ''))
        xlabel_text = f"{direction_label_str} {base_xlabel}".strip()
        
        ax.set_xlabel(xlabel_text)
        ax.set_ylabel(current_ylabel)
        
        title_parts = [f"SED Frequency Slice at {actual_freq:.2f} THz"]
        if direction_label_str:
            title_parts.append(f"({direction_label_str})")
        ax.set_title(" ".join(title_parts))
        
        if self.plot_params.get('grid', True):
            ax.grid(True, alpha=0.3)
            
        return fig, ax

    def _validate(self):
        valid_types = ['2d_intensity', '2d_phase', '3d_intensity', '3d_phase', '1d_slice', 'frequency_slice']
        if self.plot_type not in valid_types:
            # Try to check if it's a method like _plot_<type_name>
            plot_method_name = f"_plot_{self.plot_type}"
            if not (hasattr(self, plot_method_name) and callable(getattr(self, plot_method_name))):
                 raise ValueError(f"Invalid plot_type '{self.plot_type}'. Choose from {valid_types} or ensure a corresponding _plot_{self.plot_type} method exists.")
        
        # Common checks for SED object
        if not isinstance(self.sed, SED) and not (self.plot_type.startswith("3d_") and isinstance(self.sed, list)):
             raise TypeError(f"Plot type {self.plot_type} expects SED object or list for 3D, got {type(self.sed)}")

        if isinstance(self.sed, SED):
            if any(getattr(self.sed, attr, None) is None for attr in ['sed', 'freqs', 'k_points']):
                logger.warning(f"SED obj for plot {self.output_path.name} (type: {self.plot_type}) missing essential data (sed, freqs, or k_points). Plot may fail/be empty.")
        
        # Specific checks for 3D plots
        if self.plot_type.startswith('3d_'):
            if not isinstance(self.sed, list): 
                raise TypeError(f"3D plots need list of SED objects, got {type(self.sed)}")
            if not self.sed: 
                logger.warning(f"3D plot {self.output_path.name}: sed_data list is empty. No plot will be generated.")
            elif not all(isinstance(s, SED) for s in self.sed): 
                raise TypeError("3D plots: sed_data list must contain SED objects.")

    def _setup_ax_style(self, fig, ax, is_3d=False):
        theme = self.plot_params.get('theme', 'light')

        if theme == 'dark':
            plt.style.use('dark_background')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            tick_color = 'white'
            label_color = 'white'
            grid_color = 'gray'
            pane_edge_color = 'dimgray'
            line_color = 'lightgray'
            title_color = 'white'
            cbar_label_color = 'white'
            cbar_tick_color = 'white'
        else: # Default to light theme
            plt.style.use('default') # Or 'seaborn-v0_8-whitegrid'
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            tick_color = 'black'
            label_color = 'black'
            grid_color = 'lightgray' # Softer grid for light mode
            pane_edge_color = 'darkgray'
            line_color = 'black'
            title_color = 'black'
            cbar_label_color = 'black'
            cbar_tick_color = 'black'

        ax.tick_params(axis='x', colors=tick_color)
        ax.tick_params(axis='y', colors=tick_color)
        if hasattr(ax, 'xaxis'): 
            ax.xaxis.label.set_color(label_color)
        if hasattr(ax, 'yaxis'): 
            ax.yaxis.label.set_color(label_color)
        
        # Common settings for title, should be applied after specific theme colors
        # The actual title text is set in individual plot methods
        if hasattr(ax, 'title'):
             ax.title.set_color(title_color)


        if is_3d:
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor(pane_edge_color)
            ax.yaxis.pane.set_edgecolor(pane_edge_color)
            ax.zaxis.pane.set_edgecolor(pane_edge_color)
            ax.grid(True, color=grid_color, linestyle=':', linewidth=0.5, alpha=0.7 if theme == 'light' else 0.5)
            ax.xaxis.line.set_color(line_color)
            ax.yaxis.line.set_color(line_color)
            ax.zaxis.line.set_color(line_color)
            if hasattr(ax, 'zaxis'): 
                ax.tick_params(axis='z', colors=tick_color)
                ax.zaxis.label.set_color(label_color)
        else: 
            ax.grid(True, alpha=0.7 if theme == 'light' else 0.3, linestyle=':', color=grid_color)
        
        # Apply title color for all plot types, as title is set within plot methods
        # This ensures title color is consistent with the theme
        # Individual plot methods still set the text of the title
        current_title = ax.get_title()
        ax.set_title(current_title, color=title_color)
        
        # Update colorbar label and tick colors, if colorbar is created elsewhere
        # For colorbars created in _plot_2d_phase and _plot_3d, this logic needs to be inside those.
        # For pcolormesh in _plot_2d_intensity, cbar is created after this call.

    def _plot_2d_phase(self, sed_item: SED):
        if sed_item.phase is None: 
            logger.warning(f"No phase data for 2D plot: {self.output_path.name}")
            return None, None
        if sed_item.freqs is None or sed_item.k_points is None: 
            logger.warning(f"Freqs/k_points missing for phase plot {self.output_path.name}.")
            return None, None
        
        pos_mask = sed_item.freqs >= 0
        plot_f = sed_item.freqs[pos_mask]
        plot_p = sed_item.phase[pos_mask,:] if sed_item.phase.ndim==2 and sed_item.phase.shape[0]==sed_item.freqs.shape[0] else sed_item.phase
        
        if plot_f.size == 0 or sed_item.k_points.size == 0 or plot_p.size == 0:
            logger.warning(f"Not enough data for 2D phase plot {self.output_path.name}.")
            return None, None
        
        k_mesh, f_mesh = np.meshgrid(sed_item.k_points, plot_f)
        fig, ax = plt.subplots(figsize=(8,6))
        self._setup_ax_style(fig, ax)
        pcm = ax.pcolormesh(k_mesh, f_mesh, plot_p, shading='gouraud', 
                           cmap=self.plot_params['cmap'],
                           vmin=self.plot_params.get('vmin', -np.pi/2), 
                           vmax=self.plot_params.get('vmax', np.pi/2))
        
        plot_title = self.plot_params['title']
        ax.set_title(plot_title, color=self.plot_params.get('title_color', 'white' if self.plot_params.get('theme','light') == 'dark' else 'black'))
        ax.set_xlabel('k (2π/Å)')
        ax.set_ylabel('Frequency (THz)')

        max_f_plot = self.plot_params['max_freq']
        ylim_u = 1.0
        if max_f_plot is not None:
            try: 
                ylim_u_cand = float(max_f_plot)
                ylim_u = ylim_u_cand if ylim_u_cand > 0 else ylim_u
            except (ValueError, TypeError): 
                pass 
        if ylim_u == 1.0 and plot_f.size > 0: 
            ylim_u = np.max(plot_f) if np.max(plot_f) > 0 else ylim_u
        if ylim_u <= 0: 
            ylim_u = 1.0
        ax.set_ylim(0, ylim_u)

        if sed_item.k_points.size > 0: 
            ax.set_xlim(np.min(sed_item.k_points), np.max(sed_item.k_points))
        
        cbar = fig.colorbar(pcm, ax=ax, label='Phase diff (rad)')
        cbar.ax.yaxis.label.set_color(self.plot_params.get('cbar_label_color', 'white' if self.plot_params.get('theme','light') == 'dark' else 'black'))
        cbar.ax.tick_params(colors=self.plot_params.get('cbar_tick_color', 'white' if self.plot_params.get('theme','light') == 'dark' else 'black'))
        return fig, ax

    def _gather_3d_data(self, sed_list_items: List[SED], data_mode: str):
        kx_all, ky_all, freq_all, color_all = [], [], [], []
        intensity_thresh_rel = self.plot_params.get('intensity_thresh_rel_gather', 0.05)

        for sed_obj_item in sed_list_items:
            if data_mode == 'phase':
                if any(getattr(sed_obj_item, attr) is None for attr in ['phase', 'freqs', 'k_vectors']): 
                    continue
                pos_freq_mask = sed_obj_item.freqs >= 0
                plot_f_dir = sed_obj_item.freqs[pos_freq_mask]
                plot_d_dir = sed_obj_item.phase[pos_freq_mask,:] if sed_obj_item.phase.ndim==2 and sed_obj_item.phase.shape[0]==sed_obj_item.freqs.shape[0] else sed_obj_item.phase
                if not (plot_f_dir.size > 0 and sed_obj_item.k_vectors.shape[0] > 0 and plot_d_dir.size > 0 and \
                        plot_d_dir.shape[1] == sed_obj_item.k_vectors.shape[0]): 
                    continue

                for i_f, f_val in enumerate(plot_f_dir):
                    for i_k, k_v in enumerate(sed_obj_item.k_vectors):
                        if k_v.size >= 2: 
                            kx_all.append(k_v[0])
                            ky_all.append(k_v[1])
                            freq_all.append(f_val)
                            color_all.append(plot_d_dir[i_f,i_k])
            elif data_mode == 'intensity':
                kx_d, ky_d, fq_d, amp_d = sed_obj_item.gather_3d(intensity_thresh_rel=intensity_thresh_rel)
                if kx_d.size > 0: 
                    kx_all.extend(kx_d)
                    ky_all.extend(ky_d)
                    freq_all.extend(fq_d)
                    color_all.extend(amp_d)
        
        return (np.array(kx_all,dtype=np.float32), np.array(ky_all,dtype=np.float32),
                np.array(freq_all,dtype=np.float32), np.array(color_all,dtype=np.float32))

    def _plot_3d(self, kx, ky, freqs, color_data, data_mode: str):
        if kx.size == 0: 
            logger.warning(f"No data for 3D {data_mode} plot {self.output_path.name}.")
            return None, None
        
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        self._setup_ax_style(fig, ax, is_3d=True)
        
        z_data = freqs if self.plot_params.get('kz_vals') is None else self.plot_params.get('kz_vals')
        cmap, cbar_lbl = 'inferno', 'Intensity (arb.)'
        plot_c = color_data.copy()
        vmin, vmax = None, None

        if data_mode == 'phase': 
            cmap, cbar_lbl = 'coolwarm_r', 'Phase Diff (rad)'
            vmin, vmax = -np.pi/2, np.pi/2
        elif data_mode == 'intensity':
            if self.plot_params.get('intensity_log_scale',True) and plot_c.size > 0:
                pos_mask = plot_c > 1e-12
                if np.any(pos_mask): 
                    plot_c[pos_mask] = np.log10(plot_c[pos_mask])
                    plot_c[~pos_mask] = np.nan
                    cbar_lbl = 'Log10(Intensity)'
                else: 
                    logger.warning("3D intensity log scale: no positive values. Using linear.")
            
            valid_c = plot_c[~np.isnan(plot_c)]
            if valid_c.size > 0: 
                vmin, vmax = np.percentile(valid_c,5), np.percentile(valid_c,98)
            if vmax is not None and vmin is not None and vmax <= vmin: 
                vmax = vmin + (1e-9 if not np.isinf(vmin) else 1e-9)
            elif valid_c.size == 0: 
                vmin, vmax = 0, 1
        
        nan_mask = np.isnan(plot_c)
        if np.all(nan_mask) and kx.size > 0: 
            logger.warning(f"All color data NaN for 3D {data_mode} plot. Plotting gray.")
            kx_p, ky_p, z_p, c_p = kx, ky, z_data, 'gray'
            cmap = None 
        elif np.any(nan_mask):
            logger.debug(f"Filtered {np.sum(nan_mask)} NaN color points for 3D scatter.")
            kx_p, ky_p, z_p, c_p = kx[~nan_mask], ky[~nan_mask], z_data[~nan_mask], plot_c[~nan_mask]
            if kx_p.size == 0: 
                logger.warning(f"No non-NaN data for 3D {data_mode} scatter.")
                return None, None
        else: 
            kx_p, ky_p, z_p, c_p = kx, ky, z_data, plot_c

        if kx_p.size == 0: 
            logger.warning(f"No points to scatter for 3D {data_mode} plot {self.output_path.name}.")
            return None, None
            
        sc = ax.scatter(kx_p, ky_p, z_p, c=c_p, cmap=cmap, alpha=0.6, marker='o', 
                       s=15, edgecolors='none', vmin=vmin, vmax=vmax)
        ax.set_xlabel(r'$k_x$ ($2\pi/\AA$)')
        ax.set_ylabel(r'$k_y$ ($2\pi/\AA$)')
        ax.set_zlabel('Frequency (THz)' if self.plot_params.get('kz_vals') is None else r'$k_z$ ($2\pi/\AA$)')
        
        if cmap and hasattr(sc,'get_array') and sc.get_array().size > 0 and not isinstance(c_p,str): 
            cbar = plt.colorbar(sc, ax=ax, shrink=0.75, aspect=20, pad=0.1)
            cbar.set_label(cbar_lbl, color=self.plot_params.get('cbar_label_color', 'white' if self.plot_params.get('theme','light') == 'dark' else 'black'), fontsize=10)
            cbar.ax.tick_params(colors=self.plot_params.get('cbar_tick_color', 'white' if self.plot_params.get('theme','light') == 'dark' else 'black'), labelsize=9)

        plot_title_str = self.plot_params['title']
        ax.set_title(plot_title_str, color=self.plot_params.get('title_color', 'white' if self.plot_params.get('theme','light') == 'dark' else 'black'), fontsize=14)
        return fig, ax

    def generate_plot(self):
        fig = None
        if self.plot_type == '2d_intensity' and isinstance(self.sed, SED): 
            fig, _ = self._plot_2d_intensity()
        elif self.plot_type == '2d_phase' and isinstance(self.sed, SED): 
            fig, _ = self._plot_2d_phase(self.sed)
        elif self.plot_type.startswith('3d_') and isinstance(self.sed, list) and self.sed:
            mode = 'intensity' if self.plot_type == '3d_intensity' else 'phase'
            kx_all, ky_all, f_all, c_all = self._gather_3d_data(self.sed, mode)
            if kx_all.size > 0: 
                fig, _ = self._plot_3d(kx_all, ky_all, f_all, c_all, mode)
            else: 
                logger.warning(f"No data gathered for {self.plot_type} plot: {self.output_path.name}")
        
        if fig:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(self.output_path, dpi=self.plot_params.get('dpi', 300), bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Plot saved: {self.output_path.name}")
        else: 
            logger.info(f"Plot generation skipped for {self.output_path.name} (no figure/data).") 