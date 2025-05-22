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
            'heatmap_target_freq_thz': 1.0, # Default target frequency for 2D heatmap (now 3D heatmap)
            'heatmap_plane': 'xy', # Default plane for 2D heatmap (now 3D heatmap): 'xy', 'yz', or 'zx'
            'k_index': None, # For 1d_slice
            'freq_index': None, # For 1d_slice
            'highlight_region': None,
            'direction_label': '',
            'show_colorbar': True,
            'colorbar_label': 'Intensity (arb. units)',
            'grid': True,
            'tight_layout': True,
            'log_intensity': False,
            'intensity_scale': 'linear', # New parameter: 'linear', 'log', 'sqrt'
            'vmin_percentile': 0.0,
            'vmax_percentile': 100.0,
            'theme': 'light'  # Added theme parameter, default to light
        }
        
        # Update with user parameters
        self.plot_params = {**self.default_params, **kwargs}
        
    def generate_plot(self):
        self._validate() # Call validate at the beginning
        fig = None
        ax = None # Initialize ax as well

        # Apply theme settings to matplotlib's rcParams
        # This theming block seems to be from a different version, let's ensure it is present or add it.
        # For safety, I'll ensure it is present based on the earlier file read.
        theme = self.plot_params.get('theme', 'light')
        current_style_context = None # To manage style context

        if theme == 'dark':
            current_style_context = plt.style.context('dark_background')
            current_style_context.__enter__() # Apply the style
            # Further dark theme specific rcParams can be set here if needed
            # e.g., plt.rcParams['axes.facecolor'] = 'black' etc.
            # However, _setup_ax_style should handle most visual styling based on theme.
        else:
            # For light theme, we might want to ensure a default style if not dark
            # Or rely on _setup_ax_style to set colors appropriately.
            # If plt.style.use('default') is desired for light theme, it can be here.
            # current_style_context = plt.style.context('default') # Example
            # if current_style_context: current_style_context.__enter__()
            pass # Rely on _setup_ax_style for light theme specifics

        try:
            if self.plot_type == '2d_intensity' and isinstance(self.sed, SED):
                fig, ax = self._plot_2d_intensity()
            elif self.plot_type == '2d_phase' and isinstance(self.sed, SED):
                fig, ax = self._plot_2d_phase(self.sed) # Assuming sed_item is self.sed
            elif self.plot_type == '3d_heatmap' and isinstance(self.sed, SED):
                fig, ax = self._plot_3d_heatmap()
            elif self.plot_type == '1d_slice' and isinstance(self.sed, SED):
                fig, ax = self._plot_1d_slice()
            elif self.plot_type == 'frequency_slice' and isinstance(self.sed, SED):
                fig, ax = self._plot_frequency_slice()
            # Add other plot types here if they exist or are added
            # else:
            #     logger.error(f"Plot type '{self.plot_type}' is recognized but not handled in generate_plot logic.")

            if fig: # Proceed only if a figure was generated
                if self.plot_params.get('tight_layout', True):
                    fig.tight_layout()
                
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                # Use bbox_inches='tight' to prevent labels from being cut off
                fig.savefig(self.output_path, dpi=self.plot_params.get('dpi', 300), bbox_inches='tight')
                logger.info(f"Plot saved to: {self.output_path}")
            else:
                # Log if no figure was produced, e.g. due to data issues handled in plot methods
                logger.warning(f"Plot generation for {self.plot_type} did not return a figure. Output file {self.output_path} not created.")
        
        finally:
            if current_style_context:
                current_style_context.__exit__(None, None, None) # Revert style changes
            if fig: # Ensure figure is closed only if it was created
                plt.close(fig) # Close the figure to free memory
            elif ax: # If ax was created but not fig (e.g. error before fig assigned to)
                if ax.figure: plt.close(ax.figure)

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

        intensity_scale_type = self.plot_params.get('intensity_scale', 'linear').lower()
        # Backward compatibility: if log_intensity is True and intensity_scale is default linear, use log
        if self.plot_params.get('log_intensity') and intensity_scale_type == 'linear':
            intensity_scale_type = 'log'

        if intensity_scale_type == 'log':
            if np.any(intensity_to_plot > 1e-12): # Check if there's anything to log scale
                intensity_to_plot = np.log10(np.maximum(intensity_to_plot, 1e-12)) # Avoid log(0) or log(negative)
                current_colorbar_label = 'Log10(Intensity)'
            else:
                logger.warning("Log scaling requested for intensity, but all values are too small or zero. Using linear scale.")
        elif intensity_scale_type == 'sqrt':
            if np.any(intensity_to_plot >= 0): # Check if there's anything to sqrt scale
                intensity_to_plot = np.sqrt(np.maximum(intensity_to_plot, 0)) # Avoid sqrt(negative)
                current_colorbar_label = 'Sqrt(Intensity)'
            else:
                logger.warning("Sqrt scaling requested for intensity, but all values are negative. Using linear scale.")
        elif intensity_scale_type == 'dsqrt':
            if np.any(intensity_to_plot >= 0):
                intensity_to_plot = np.sqrt(np.sqrt(np.maximum(intensity_to_plot, 0))) # Avoid sqrt(negative)
                current_colorbar_label = 'DSqrt(Intensity)'
            else:
                logger.warning("DSqrt scaling requested for intensity, but all values are negative. Using linear scale.")
        elif intensity_scale_type != 'linear':
            logger.warning(f"Unknown intensity_scale_type '{intensity_scale_type}'. Using linear scale.")

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
        # OriginalPSA adds direction_label to title if not used for xlabel. Here, direction_label makes xlabel specific.
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
                if self.plot_params.get('highlight_label', False): # OriginalPSA has show_target_label
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
        
        intensity_scale_type = self.plot_params.get('intensity_scale', 'linear').lower()
        # Backward compatibility
        if self.plot_params.get('log_intensity') and intensity_scale_type == 'linear':
            intensity_scale_type = 'log'

        if intensity_scale_type == 'log':
            if np.any(intensity_data > 1e-12):
                intensity_data = np.log10(np.maximum(intensity_data, 1e-12))
                ylabel = 'Log10(Intensity)'
            else:
                logger.warning("Log scaling requested for intensity, but all values are too small or zero. Using linear scale.")
        elif intensity_scale_type == 'sqrt':
            if np.any(intensity_data >= 0):
                intensity_data = np.sqrt(np.maximum(intensity_data, 0))
                ylabel = 'Sqrt(Intensity)'
            else:
                logger.warning("Sqrt scaling requested for intensity, but all values are negative. Using linear scale.")
        elif intensity_scale_type == 'dsqrt':
            if np.any(intensity_data >= 0):
                intensity_data = np.sqrt(np.sqrt(np.maximum(intensity_data, 0)))
                ylabel = 'DSqrt(Intensity)'
            else:
                logger.warning("DSqrt scaling requested for intensity, but all values are negative. Using linear scale.")
        elif intensity_scale_type != 'linear':
            logger.warning(f"Unknown intensity_scale_type '{intensity_scale_type}'. Using linear scale.")

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
        intensity_scale_type = self.plot_params.get('intensity_scale', 'linear').lower()
        # Backward compatibility
        if self.plot_params.get('log_intensity') and intensity_scale_type == 'linear':
            intensity_scale_type = 'log'
        
        if intensity_scale_type == 'log':
            if np.any(plot_data > 1e-12):
                plot_data = np.log10(np.maximum(plot_data, 1e-12))
                current_ylabel = 'Log10(Intensity)'
            else:
                logger.warning("Log scaling requested for intensity, but all values are too small or zero. Using linear scale.")
        elif intensity_scale_type == 'sqrt':
            if np.any(plot_data >= 0):
                plot_data = np.sqrt(np.maximum(plot_data, 0))
                current_ylabel = 'Sqrt(Intensity)'
            else:
                logger.warning("Sqrt scaling requested for intensity, but all values are negative. Using linear scale.")
        elif intensity_scale_type == 'dsqrt':
            if np.any(plot_data >= 0):
                plot_data = np.sqrt(np.sqrt(np.maximum(plot_data, 0)))
                current_ylabel = 'DSqrt(Intensity)'
            else:
                logger.warning("DSqrt scaling requested for intensity, but all values are negative. Using linear scale.")
        elif intensity_scale_type != 'linear':
            logger.warning(f"Unknown intensity_scale_type '{intensity_scale_type}'. Using linear scale.")

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
        valid_types = ['2d_intensity', '2d_phase', '1d_slice', 'frequency_slice', '3d_heatmap']
        if self.plot_type not in valid_types:
            # Try to check if it's a method like _plot_<type_name>
            plot_method_name = f"_plot_{self.plot_type}"
            if not (hasattr(self, plot_method_name) and callable(getattr(self, plot_method_name))):
                 raise ValueError(f"Invalid plot_type '{self.plot_type}'. Choose from {valid_types} or ensure a corresponding _plot_{self.plot_type} method exists.")
        
        # Common checks for SED object
        if not isinstance(self.sed, SED):
             raise TypeError(f"Plot type {self.plot_type} expects SED object, got {type(self.sed)}")

        if isinstance(self.sed, SED):
            if any(getattr(self.sed, attr, None) is None for attr in ['sed', 'freqs', 'k_points', 'k_vectors']):
                logger.warning(f"SED obj for plot {self.output_path.name} (type: {self.plot_type}) missing essential data (sed, freqs, k_points, or k_vectors). Plot may fail/be empty.")
            if self.plot_type == '3d_heatmap':
                if getattr(self.sed, 'k_grid_shape', None) is None or not isinstance(self.sed.k_grid_shape, tuple) or len(self.sed.k_grid_shape) != 2:
                    raise ValueError("For '3d_heatmap', SED.k_grid_shape must be a 2-tuple (e.g., (nkx, nky)).")
                plane = self.plot_params.get('heatmap_plane', 'xy').lower()
                if plane not in ['xy', 'yz', 'zx']:
                    raise ValueError(f"Invalid 'heatmap_plane': {plane}. Must be 'xy', 'yz', or 'zx'.")

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
        else: # For 2D plots, respect the 'grid' parameter, defaulting to True if not specified for general 2D plots.
            if self.plot_params.get('grid', True): # Default to True for general 2D plots
                ax.grid(True, alpha=0.7 if theme == 'light' else 0.3, linestyle=':', color=grid_color)
            else:
                ax.grid(False) # Explicitly turn off if grid=False
        
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

    def _plot_3d_heatmap(self) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """
        Plots a 2D heatmap of SED intensity on a specified k-plane (xy, yz, or zx)
        at a selected frequency.
        """
        fig, ax = plt.subplots(figsize=self.plot_params.get('figsize', (8, 6.5)))
        self._setup_ax_style(fig, ax, is_3d=False) # Heatmap is 2D
        ax.grid(False) # Ensure grid is off by default for heatmaps

        if self.sed.k_grid_shape is None or len(self.sed.k_grid_shape) != 2:
            logger.error("SED.k_grid_shape is not valid for 3D heatmap.")
            plt.close(fig)
            return None, None

        target_freq_thz = self.plot_params.get('heatmap_target_freq_thz', 1.0)
        plane = self.plot_params.get('heatmap_plane', 'xy').lower()

        # Find the closest frequency index
        if self.sed.freqs is None or self.sed.freqs.size == 0:
            logger.error("SED object has no frequency data for 3D heatmap.")
            plt.close(fig)
            return None, None
        
        freq_idx = np.argmin(np.abs(self.sed.freqs - target_freq_thz))
        actual_freq = self.sed.freqs[freq_idx]
        logger.info(f"Plotting 3D heatmap for frequency {actual_freq:.3f} THz (target was {target_freq_thz:.3f} THz).")

        # Calculate intensity for the selected frequency slice
        if self.sed.is_complex:
            intensity_at_freq = np.sum(np.abs(self.sed.sed[freq_idx, :, :])**2, axis=-1)
        else:
            if self.sed.sed.ndim == 3: # (n_freqs, n_kpoints, n_polarizations_summed)
                 intensity_at_freq = np.sum(self.sed.sed[freq_idx, :, :], axis=-1)
            elif self.sed.sed.ndim == 2: # (n_freqs, n_kpoints) - already intensity
                 intensity_at_freq = self.sed.sed[freq_idx, :]
            else:
                logger.error(f"Unsupported SED data format for 3D heatmap: ndim={self.sed.sed.ndim}")
                plt.close(fig)
                return None, None
        
        if intensity_at_freq.size != self.sed.k_vectors.shape[0]:
            logger.error(f"Intensity data size ({intensity_at_freq.size}) does not match number of k-vectors ({self.sed.k_vectors.shape[0]}).")
            plt.close(fig)
            return None, None

        # Reshape intensity to grid
        n_kx, n_ky = self.sed.k_grid_shape
        if intensity_at_freq.size != n_kx * n_ky:
            logger.error(f"Intensity data size ({intensity_at_freq.size}) does not match k_grid_shape ({n_kx}x{n_ky}={n_kx*n_ky}).")
            plt.close(fig)
            return None, None
        intensity_grid = intensity_at_freq.reshape(self.sed.k_grid_shape)

        # Extract k-components for the specified plane
        k_vectors_flat = self.sed.k_vectors # Shape (nkx*nky, 3)
        if plane == "xy":
            k_comp1_flat = k_vectors_flat[:, 0]
            k_comp2_flat = k_vectors_flat[:, 1]
            xlabel = r'$k_x$ ($2\pi/\AA$)'
            ylabel = r'$k_y$ ($2\pi/\AA$)'
        elif plane == "yz":
            k_comp1_flat = k_vectors_flat[:, 1]
            k_comp2_flat = k_vectors_flat[:, 2]
            xlabel = r'$k_y$ ($2\pi/\AA$)'
            ylabel = r'$k_z$ ($2\pi/\AA$)'
        elif plane == "zx":
            k_comp1_flat = k_vectors_flat[:, 2] # k_comp1 is z
            k_comp2_flat = k_vectors_flat[:, 0] # k_comp2 is x
            xlabel = r'$k_z$ ($2\pi/\AA$)'
            ylabel = r'$k_x$ ($2\pi/\AA$)'
        else: # Should be caught by _validate, but as a safeguard
            logger.error(f"Invalid plane '{plane}' in _plot_3d_heatmap.")
            plt.close(fig)
            return None, None

        # Create meshgrid for k-components. We need unique sorted values for contourf/pcolormesh.
        # The k_vectors are generated by iterating kx then ky (for xy plane).
        # So, k_comp1_flat.reshape(n_kx, n_ky) would have kx varying along columns, ky along rows.
        # k_comp1_vals = np.unique(k_comp1_flat) # Should be n_kx unique values
        # k_comp2_vals = np.unique(k_comp2_flat) # Should be n_ky unique values

        # For pcolormesh, X and Y define the corners of the cells.
        # If k_comp1_flat was, e.g., kx from SEDCalculator.get_k_grid (plane='xy')
        # kx_vals = np.linspace(k_range_x[0], k_range_x[1], n_kx)
        # ky_vals = np.linspace(k_range_y[0], k_range_y[1], n_ky)
        # k_vectors_list.append([kx, ky, k_fixed_val]) for kx in kx_vals for ky in ky_vals
        # This means k_vectors_flat[:,0] = np.tile(kx_vals, n_ky) which is not what we want for meshgrid directly.
        # And k_vectors_flat[:,1] = np.repeat(ky_vals, n_kx)
        
        # Let's get the unique sorted kx and ky values that form the grid axes
        # Based on how get_k_grid creates k_vectors_list for 'xy':
        # kx changes fastest in the flattened k_vectors_list if the inner loop is ky:
        #   for kx in kx_vals: for ky in ky_vals: append([kx,ky,kz]) -> k_vectors_3d.reshape(nkx,nky,3)
        #   then k_vectors_3d[:,:,0] is kx, k_vectors_3d[:,:,1] is ky
        # If the loops are: for kx in kx_vals: for ky in ky_vals:
        #   k_comp1_flat (kx) will be [kx0, kx0, ..., kx0 (nky times), kx1, kx1, ...]
        #   k_comp2_flat (ky) will be [ky0, ky1, ..., ky(nky-1), ky0, ky1, ...]
        
        # Let's reconstruct the axis vectors
        k1_axis = np.unique(k_comp1_flat) # e.g. unique kx values
        k2_axis = np.unique(k_comp2_flat) # e.g. unique ky values

        if len(k1_axis) != n_kx or len(k2_axis) != n_ky:
             logger.warning(f"Mismatch in unique k-component values and k_grid_shape. Expected ({n_kx}, {n_ky}), got ({len(k1_axis)}, {len(k2_axis)}). This might affect plot axes.")
             # Fallback, assuming they are sorted correctly from linspace if unique fails
             if len(k1_axis) != n_kx: k1_axis = np.linspace(k_comp1_flat.min(), k_comp1_flat.max(), n_kx)
             if len(k2_axis) != n_ky: k2_axis = np.linspace(k_comp2_flat.min(), k_comp2_flat.max(), n_ky)


        K1, K2 = np.meshgrid(k1_axis, k2_axis) # K1 will be kx, K2 will be ky if plane is xy

        # The intensity_grid is (n_kx, n_ky).
        # If K1 from meshgrid(kx_axis, ky_axis) has shape (n_ky, n_kx)
        # and K2 has shape (n_ky, n_kx)
        # then intensity_grid needs to be transposed if it's (n_kx, n_ky) for pcolormesh.
        # Let's check pcolormesh docs: C has to be (ny, nx) if X,Y are (ny,nx) or (ny+1,nx+1)
        # Our meshgrid(k1_axis, k2_axis) makes K1, K2 shapes (len(k2_axis), len(k1_axis))
        # i.e. (n_ky, n_kx)
        # So, intensity_grid, which is (n_kx, n_ky), needs to be intensity_grid.T for pcolormesh.
        
        plot_intensity_data = intensity_grid.T # Transpose to match meshgrid (n_ky, n_kx)

        current_colorbar_label = self.plot_params['colorbar_label']
        intensity_scale_type = self.plot_params.get('intensity_scale', 'linear').lower()
        # Backward compatibility
        if self.plot_params.get('log_intensity') and intensity_scale_type == 'linear':
            intensity_scale_type = 'log'

        if intensity_scale_type == 'log':
            if np.any(plot_intensity_data > 1e-12):
                plot_intensity_data = np.log10(np.maximum(plot_intensity_data, 1e-12))
                current_colorbar_label = 'Log10(Intensity)'
            else:
                logger.warning("Log scaling requested, but all values too small. Using linear scale.")
        elif intensity_scale_type == 'sqrt':
            if np.any(plot_intensity_data >= 0): # Check if there's anything to sqrt scale
                plot_intensity_data = np.sqrt(np.maximum(plot_intensity_data, 0)) # Avoid sqrt(negative)
                current_colorbar_label = 'Sqrt(Intensity)'
            else:
                logger.warning("Sqrt scaling requested for intensity, but all values are negative. Using linear scale.")
        elif intensity_scale_type == 'dsqrt':
            if np.any(plot_intensity_data >= 0):
                plot_intensity_data = np.sqrt(np.sqrt(np.maximum(plot_intensity_data, 0)))
                current_colorbar_label = 'DSqrt(Intensity)'
            else:
                logger.warning("DSqrt scaling requested for intensity, but all values are negative. Using linear scale.")
        elif intensity_scale_type != 'linear':
            logger.warning(f"Unknown intensity_scale_type '{intensity_scale_type}'. Using linear scale.")

        # Determine vmin and vmax for the color scale
        # Prioritize vmin/vmax if directly provided in plot_params
        vmin = self.plot_params.get('vmin')
        vmax = self.plot_params.get('vmax')

        if vmin is None or vmax is None: # If not directly provided, calculate from percentiles of current slice
            valid_intensity_values = plot_intensity_data[~np.isnan(plot_intensity_data) & ~np.isinf(plot_intensity_data)]
            if valid_intensity_values.size > 0:
                calculated_vmin = np.percentile(valid_intensity_values, self.plot_params.get('vmin_percentile', 0.0))
                calculated_vmax = np.percentile(valid_intensity_values, self.plot_params.get('vmax_percentile', 100.0))
                
                if calculated_vmin == calculated_vmax: # Handle flat data
                    calculated_vmin = calculated_vmin - 0.1 if calculated_vmin != 0 else -0.1
                    calculated_vmax = calculated_vmax + 0.1 if calculated_vmax != 0 else 0.1
                
                if vmin is None:
                    vmin = calculated_vmin
                if vmax is None:
                    vmax = calculated_vmax
            else: # No valid data to calculate percentiles, use safe defaults if vmin/vmax still None
                if vmin is None: vmin = 0
                if vmax is None: vmax = 1
        
        pcm = ax.pcolormesh(K1, K2, plot_intensity_data,
                            cmap=self.plot_params['cmap'],
                            shading='gouraud',
                            vmin=vmin, vmax=vmax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title = self.plot_params.get('title', 'SED Heatmap')
        ax.set_title(f"{title} @ {actual_freq:.2f} THz (Plane: {plane.upper()})")

        if self.plot_params['show_colorbar'] and hasattr(pcm, 'get_array') and pcm.get_array().size > 0:
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label(current_colorbar_label)
        
        # Explicitly control grid based on plot_params, defaulting to False for heatmaps
        if self.plot_params.get('grid', False): # Default to False for heatmaps
            ax.grid(True, alpha=0.3, linestyle=':')

        ax.set_aspect('equal', adjustable='box') # Make kx and ky scales equal if appropriate

        return fig, ax 