#!/usr/bin/env python3
"""
PSA GUI - Interactive Phonon Spectral Analysis Interface

This GUI provides:
- Trajectory file upload and parameter specification
- Interactive SED plotting with clickable points
- Automatic iSED reconstruction and atomic motion visualization
- Real-time parameter adjustment
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
import logging
from pathlib import Path
import sys
from typing import Optional, Tuple, Dict, Any
import tempfile
import subprocess
import os
import traceback
import ast
from mpl_toolkits.axes_grid1 import make_axes_locatable # For better colorbar placement
import imageio  # For GIF creation
from datetime import datetime
from io import BytesIO
from PIL import Image

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

from psa import TrajectoryLoader, SEDCalculator, SED, SEDPlotter

# Try to import PIL for image handling, but make it optional
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolTip:
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip) # Delay in ms

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert") # Get size of widget
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # Creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffe0", relief='solid', borderwidth=1,
                       font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()

class ProgressDialog:
    """Simple progress dialog for trajectory loading"""
    def __init__(self, parent, title="Loading..."):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (150 // 2)
        self.dialog.geometry(f"400x150+{x}+{y}")
        
        # Create progress elements
        self.label = ttk.Label(self.dialog, text="Initializing...", font=("Arial", 10))
        self.label.pack(pady=20)
        
        self.progress = ttk.Progressbar(self.dialog, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill=tk.X)
        self.progress.start()
        
        self.detail_label = ttk.Label(self.dialog, text="", font=("Arial", 8))
        self.detail_label.pack(pady=5)
        
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close button
        
    def update_message(self, message, detail=""):
        self.label.configure(text=message)
        self.detail_label.configure(text=detail)
        self.dialog.update_idletasks()
        
    def close(self):
        self.progress.stop()
        self.dialog.destroy()

class PSAMainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("PSA - Phonon Spectral Analysis GUI")
        self.root.geometry("1400x900")
        
        # Initialize state variables
        self.trajectory_file = None
        self.sed_calculator = None
        self.sed_result = None
        self.current_plot_data = None
        self.ised_result_path = None
        self.click_marker = None  # Store the click marker
        self.sed_colorbar = None  # Initialize colorbar reference
        self.animation_running = False  # Track animation state
        self.current_temp_ised_dir = None # Path to temp dir for iSED output
        
        # Create main interface
        self._create_interface()
        
        # Bind events
        self._setup_event_handlers()
    
    def _cleanup_colorbar(self):
        """Helper method to clean up existing colorbar and its axes"""
        try:
            if hasattr(self, 'sed_colorbar') and self.sed_colorbar:
                self.sed_colorbar.remove()
                self.sed_colorbar = None
            if hasattr(self, 'sed_colorbar_ax') and self.sed_colorbar_ax:
                if self.sed_colorbar_ax.get_figure() == self.sed_fig:
                    self.sed_colorbar_ax.remove()
                self.sed_colorbar_ax = None
        except Exception:
            pass
    
    def _create_labeled_entry(self, parent, label_text, textvariable, pack_kwargs=None, entry_kwargs=None, label_kwargs=None):
        """Helper method to create label + entry combinations"""
        pack_kwargs = pack_kwargs or {}
        entry_kwargs = entry_kwargs or {}
        label_kwargs = label_kwargs or {}
        
        label = ttk.Label(parent, text=label_text, **label_kwargs)
        label.pack(anchor="w", **pack_kwargs)
        
        entry = ttk.Entry(parent, textvariable=textvariable, **entry_kwargs)
        entry.pack(fill="x", pady=(0,5))
        
        return label, entry
    
    def _create_labeled_frame_with_entry(self, parent, label_text, textvariable, button_text=None, button_command=None):
        """Helper method to create frame with label, entry, and optional button"""
        ttk.Label(parent, text=label_text, font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0,10))
        
        entry = ttk.Entry(frame, textvariable=textvariable, state="readonly" if button_text else "normal")
        entry.pack(side="left", fill="x", expand=True)
        
        if button_text and button_command:
            button = ttk.Button(frame, text=button_text, command=button_command)
            button.pack(side="right", padx=(5,0))
            return frame, entry, button
        
        return frame, entry

    def _create_interface(self):
        """Create the main GUI interface"""
        # Create main paned window with visible separator
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, 
                                  sashwidth=8, sashrelief=tk.RAISED, 
                                  bg='lightgray', sashpad=2)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        self._create_control_panel(main_pane)
        
        # Right panel for plots
        self._create_plot_panel(main_pane)
        
    def _create_control_panel(self, parent):
        """Create the left control panel"""
        control_frame = ttk.Frame(parent, width=450)
        control_frame.pack_propagate(False)  # Maintain minimum width
        parent.add(control_frame, minsize=400)  # Set minimum size for the pane
        
        # Create notebook for organized tabs
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File Input Tab
        self._create_file_tab(notebook)
        
        # SED Parameters Tab (now combined with k-grid)
        self._create_sed_tab(notebook)
        
        # Visualization Tab
        self._create_viz_tab(notebook)
        
        # iSED Tab
        self._create_ised_tab(notebook)
        
        # Remove k-grid tab as it's now combined with SED Parameters

    def _create_file_tab(self, notebook):
        """Create file input and basic parameters tab"""
        file_frame = ttk.Frame(notebook)
        notebook.add(file_frame, text="I/O")
        
        # Trajectory file selection
        self.trajectory_var = tk.StringVar()
        traj_frame, trajectory_entry, browse_button = self._create_labeled_frame_with_entry(
            file_frame, "Trajectory File:", self.trajectory_var, "Browse...", self._browse_trajectory)
        
        # File format selection
        ttk.Label(file_frame, text="File Format:").pack(anchor="w", pady=(5,0))
        self.format_var = tk.StringVar(value="lammps")
        format_combo = ttk.Combobox(file_frame, textvariable=self.format_var, 
                                   values=["lammps", "xyz", "auto"], state="readonly")
        format_combo.pack(fill="x", pady=(0,10))
        
        # MD parameters
        ttk.Label(file_frame, text="MD Parameters:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        # Timestep
        self.dt_var = tk.DoubleVar(value=0.005)
        self._create_labeled_entry(file_frame, "Timestep (ps):", self.dt_var)
        
        # System dimensions
        dims_frame = ttk.Frame(file_frame)
        dims_frame.pack(fill="x", pady=(5,10))
        ttk.Label(dims_frame, text="System Dimensions:").pack(anchor="w")
        dim_row = ttk.Frame(dims_frame)
        dim_row.pack(fill="x")
        
        # Create dimension entries in a more compact way
        self.nx_var = tk.IntVar(value=50)
        self.ny_var = tk.IntVar(value=50)
        self.nz_var = tk.IntVar(value=1)
        dims = [("nx:", self.nx_var), ("ny:", self.ny_var), ("nz:", self.nz_var)]
        
        for label_text, var in dims:
            ttk.Label(dim_row, text=label_text).pack(side="left")
            ttk.Entry(dim_row, textvariable=var, width=8).pack(side="left", padx=(2, 10))
        
        # Load trajectory button
        self.load_button = ttk.Button(file_frame, text="Load Trajectory", 
                                     command=self._load_trajectory, state="disabled")
        self.load_button.pack(fill="x", pady=(20,10))
        
        # Status
        self.status_var = tk.StringVar(value="No trajectory loaded")
        ttk.Label(file_frame, textvariable=self.status_var, foreground="blue").pack(anchor="w")
        
        # Output directory selection
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "psa_results"))
        output_dir_frame, self.output_dir_entry, output_browse_button = self._create_labeled_frame_with_entry(
            file_frame, "Output Directory:", self.output_dir_var, "Browse...", self._browse_output_dir)
        
        ttk.Label(file_frame, text="All saved files will be stored in this directory", 
                 font=("Arial", 8), foreground="gray").pack(anchor="w")
        
    def _create_sed_tab(self, notebook):
        """Create SED calculation parameters tab (now includes k-grid)"""
        sed_frame = ttk.Frame(notebook, padding=(10, 10))
        notebook.add(sed_frame, text="Calculation")
        
        # Calculation Mode Toggle
        ttk.Label(sed_frame, text="Calculation Mode:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        self.calc_mode_var = tk.StringVar(value="K-Path")
        calc_mode_frame = ttk.Frame(sed_frame)
        calc_mode_frame.pack(fill="x", pady=(0,10))
        ttk.Radiobutton(calc_mode_frame, text="K-Path", variable=self.calc_mode_var, value="K-Path", command=self._toggle_calc_mode).pack(side="left", padx=(0,20))
        ttk.Radiobutton(calc_mode_frame, text="K-Grid", variable=self.calc_mode_var, value="K-Grid", command=self._toggle_calc_mode).pack(side="left")
        
        # K-Path Parameters Frame
        self.kpath_frame = ttk.LabelFrame(sed_frame, text="K-Path Parameters")
        
        # K-path Direction specification
        ttk.Label(self.kpath_frame, text="K-path Direction:").pack(anchor="w", pady=(5,0))
        self.direction_var = tk.StringVar(value="[1,1,0]")
        direction_entry = ttk.Entry(self.kpath_frame, textvariable=self.direction_var)
        direction_entry.pack(fill="x", pady=(0,10))
        ToolTip(direction_entry, text="Specify k-path direction, e.g., [1,0,0], 'x', or 'xy'.\nFor 3D vectors like [h,k,l], ensure values are integers or floats.")
        
        # Number of k-points
        ttk.Label(self.kpath_frame, text="Number of k-points:").pack(anchor="w")
        self.n_k_var = tk.IntVar(value=250)
        validate_int_cmd = (self.kpath_frame.register(self._validate_int_input), '%P')
        nk_entry = ttk.Entry(self.kpath_frame, textvariable=self.n_k_var, validate='key', validatecommand=validate_int_cmd)
        nk_entry.pack(fill="x", pady=(0,5))
        ToolTip(nk_entry, text="Number of k-points along the specified k-path (integer > 0).")
        
        # BZ coverage
        ttk.Label(self.kpath_frame, text="Reciprocal Space Coverage:").pack(anchor="w")
        self.bz_coverage_var = tk.DoubleVar(value=4.0)
        bz_coverage_entry = ttk.Entry(self.kpath_frame, textvariable=self.bz_coverage_var)
        bz_coverage_entry.pack(fill="x", pady=(0,10))
        ToolTip(bz_coverage_entry, text="Coverage factor for reciprocal space extent in the k-direction.\n"
                                        "Uses directional projection onto reciprocal lattice vectors.\n"
                                        "1.0 = Γ to BZ boundary, 4.0 = 4× BZ boundary (typical).")
        
        # K-Grid Parameters Frame
        self.kgrid_frame = ttk.LabelFrame(sed_frame, text="K-Grid Parameters")
        
        # Plane selector
        plane_frame = ttk.Frame(self.kgrid_frame)
        plane_frame.pack(fill="x", pady=(5,5))
        ttk.Label(plane_frame, text="Plane:").pack(side="left")
        self.kgrid_plane_var = tk.StringVar(value="xy")
        plane_combo = ttk.Combobox(plane_frame, textvariable=self.kgrid_plane_var, values=["xy", "yz", "zx"], state="readonly", width=6)
        plane_combo.pack(side="left", padx=(5,0))
        plane_combo.bind("<<ComboboxSelected>>", lambda event: self._update_kgrid_axis_controls())
        
        # K-grid axis controls (will be populated by _update_kgrid_axis_controls)
        self.kgrid_kx_min_var = tk.DoubleVar(value=-1.0)
        self.kgrid_kx_max_var = tk.DoubleVar(value=1.0)
        self.kgrid_ky_min_var = tk.DoubleVar(value=-1.0)
        self.kgrid_ky_max_var = tk.DoubleVar(value=1.0)
        self.kgrid_kz_val_var = tk.DoubleVar(value=0.0)
        
        # Create axis control widgets
        axis_grid_frame = ttk.Frame(self.kgrid_frame)
        axis_grid_frame.pack(fill="x", pady=(5,10))
        
        self.kgrid_axis_labels = {}
        self.kgrid_axis_entries = {}
        # kx min/max
        self.kgrid_axis_labels['kx_min'] = ttk.Label(axis_grid_frame, text="kx min:")
        self.kgrid_axis_entries['kx_min'] = ttk.Entry(axis_grid_frame, textvariable=self.kgrid_kx_min_var, width=8)
        self.kgrid_axis_labels['kx_max'] = ttk.Label(axis_grid_frame, text="kx max:")
        self.kgrid_axis_entries['kx_max'] = ttk.Entry(axis_grid_frame, textvariable=self.kgrid_kx_max_var, width=8)
        # ky min/max
        self.kgrid_axis_labels['ky_min'] = ttk.Label(axis_grid_frame, text="ky min:")
        self.kgrid_axis_entries['ky_min'] = ttk.Entry(axis_grid_frame, textvariable=self.kgrid_ky_min_var, width=8)
        self.kgrid_axis_labels['ky_max'] = ttk.Label(axis_grid_frame, text="ky max:")
        self.kgrid_axis_entries['ky_max'] = ttk.Entry(axis_grid_frame, textvariable=self.kgrid_ky_max_var, width=8)
        # kz (fixed)
        self.kgrid_axis_labels['kz_fixed'] = ttk.Label(axis_grid_frame, text="kz (fixed):")
        self.kgrid_axis_entries['kz_fixed'] = ttk.Entry(axis_grid_frame, textvariable=self.kgrid_kz_val_var, width=8)
        
        # Grid size
        grid_size_frame = ttk.Frame(self.kgrid_frame)
        grid_size_frame.pack(fill="x", pady=(0,10))
        self.kgrid_n_kx_label = ttk.Label(grid_size_frame, text="n_kx:")
        self.kgrid_n_kx_label.pack(side="left")
        self.kgrid_n_kx_var = tk.IntVar(value=40)
        ttk.Entry(grid_size_frame, textvariable=self.kgrid_n_kx_var, width=8).pack(side="left", padx=(5,15))
        self.kgrid_n_ky_label = ttk.Label(grid_size_frame, text="n_ky:")
        self.kgrid_n_ky_label.pack(side="left")
        self.kgrid_n_ky_var = tk.IntVar(value=40)
        ttk.Entry(grid_size_frame, textvariable=self.kgrid_n_ky_var, width=8).pack(side="left", padx=(5,0))
        
        # Common Parameters (always shown)
        common_frame = ttk.LabelFrame(sed_frame, text="Common Parameters")
        common_frame.pack(fill="x", pady=(10,0))
        
        # Basis atoms
        ttk.Label(common_frame, text="Basis Atom Types (comma-separated, empty for all):").pack(anchor="w", pady=(5,0))
        self.basis_types_var = tk.StringVar()
        basis_types_entry = ttk.Entry(common_frame, textvariable=self.basis_types_var)
        basis_types_entry.pack(fill="x", pady=(0,10))
        ToolTip(basis_types_entry, text="Comma-separated list of atom types to include in SED calculation (e.g., '1,2').\nLeave empty to use all atom types.")

        # Summation mode
        ttk.Label(common_frame, text="Summation Mode:").pack(anchor="w", pady=(5,0))
        self.summation_mode_var = tk.StringVar(value="coherent")
        summation_mode_combo = ttk.Combobox(common_frame, textvariable=self.summation_mode_var,
                                   values=["coherent", "incoherent"], state="readonly")
        summation_mode_combo.pack(fill="x", pady=(0,10))
        ToolTip(summation_mode_combo, text="Mode for summing atomic contributions to SED:\n- coherent: Sum complex amplitudes (default)\n- incoherent: Sum squared magnitudes")
        
        # Chirality options (available for both K-Path and K-Grid) - moved to common parameters
        self.chiral_sed_var = tk.BooleanVar(value=False)
        chiral_check = ttk.Checkbutton(common_frame, text="Calculate Chirality", variable=self.chiral_sed_var, command=self._toggle_chiral_options)
        chiral_check.pack(anchor="w", pady=(5,0))
        ToolTip(chiral_check, text="Enable to calculate chirality (requires coherent summation for both K-Path and K-Grid).")

        self.chiral_options_frame = ttk.Frame(common_frame)
        ttk.Label(self.chiral_options_frame, text="Chiral Axis:").pack(anchor="w", pady=(5,0))
        self.chiral_axis_var = tk.StringVar(value="z")
        self.chiral_axis_combo = ttk.Combobox(self.chiral_options_frame, textvariable=self.chiral_axis_var,
                                              values=["x", "y", "z"], state="readonly", width=8)
        self.chiral_axis_combo.pack(fill="x", pady=(0,5))
        ToolTip(self.chiral_axis_combo, text="Select the chiral axis. The phase will be calculated using the two orthogonal polarizations.")
        
        # Set initial mode
        self._update_kgrid_axis_controls()
        
        # Spacer to push Calculate SED button to bottom
        self.sed_spacer_frame = ttk.Frame(sed_frame)
        self.sed_spacer_frame.pack(fill="both", expand=True)
        
        # Now set the calculation mode (after spacer frame exists)
        self._toggle_calc_mode()
        
        # Calculate SED button (moved to bottom for better UX)
        self.calc_sed_button = ttk.Button(sed_frame, text="Calculate SED", 
                                         command=self._calculate_sed, state="disabled")
        self.calc_sed_button.pack(fill="x", pady=(30,10), side="bottom")
        
        # SED status
        self.sed_status_var = tk.StringVar(value="No SED calculated")
        ttk.Label(sed_frame, textvariable=self.sed_status_var, foreground="blue").pack(anchor="w", side="bottom")

    def _create_viz_tab(self, notebook):
        """Create visualization parameters tab"""
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Plot")
        
        # Use the viz_frame directly as the parent for all plot options (no scrolling)
        plot_content_frame = viz_frame

        # Max frequency (shared for both K-Path and K-Grid)
        ttk.Label(plot_content_frame, text="Max Frequency (THz, empty for auto):").pack(anchor="w", pady=(5,0))
        self.max_freq_var = tk.StringVar()
        max_freq_entry = ttk.Entry(plot_content_frame, textvariable=self.max_freq_var)
        max_freq_entry.pack(fill="x", pady=(0,5))
        max_freq_entry.bind('<Return>', lambda event: self._on_max_freq_change())
        max_freq_entry.bind('<FocusOut>', lambda event: self._on_max_freq_change())
        ToolTip(max_freq_entry, text="Maximum frequency (THz) to display on plots.\nLeave empty for automatic scaling based on data.")

        # Plot Chiral toggle button (only shown when chiral data is available) - BEFORE save options
        self.plot_chiral_frame = ttk.Frame(plot_content_frame)
        # Don't pack initially - will be shown by _update_viz_controls_state when chiral data is available
        self.plot_chiral_var = tk.BooleanVar(value=False)
        self.plot_chiral_button = ttk.Checkbutton(self.plot_chiral_frame, text="Plot Chiral", 
                                                 variable=self.plot_chiral_var, 
                                                 command=self._on_plot_chiral_toggle)
        self.plot_chiral_button.pack(anchor="w", pady=(0,10))
        ToolTip(self.plot_chiral_button, text="Toggle to display chirality instead of SED intensity.")

        # Intensity scaling (shown for both modes)
        ttk.Label(plot_content_frame, text="Intensity Scaling:").pack(anchor="w")
        self.intensity_scale_var = tk.StringVar(value="dsqrt")
        scale_combo = ttk.Combobox(plot_content_frame, textvariable=self.intensity_scale_var,
                                  values=["linear", "log", "sqrt", "dsqrt"], state="readonly")
        scale_combo.pack(fill="x", pady=(0,10))
        self.intensity_scale_combo = scale_combo
        self.intensity_scale_combo.bind("<<ComboboxSelected>>", lambda event: self._on_intensity_scale_change())

        # Intensity Colormap (adapt based on mode)
        ttk.Label(plot_content_frame, text="Intensity Colormap:").pack(anchor="w")
        self.colormap_var = tk.StringVar(value="inferno")
        cmap_combo = ttk.Combobox(plot_content_frame, textvariable=self.colormap_var,
                                 values=["inferno", "viridis", "plasma", "magma", "hot", "jet"], 
                                 state="readonly")
        cmap_combo.pack(fill="x", pady=(0,10))
        ToolTip(cmap_combo, text="Colormap for SED intensity plot.")
        self.intensity_colormap_combo = cmap_combo
        self.intensity_colormap_combo.bind("<<ComboboxSelected>>", lambda event: self._on_intensity_colormap_change())

        # Phase Colormap (only shown when chiral toggle is enabled)
        self.phase_colormap_frame = ttk.Frame(plot_content_frame)
        # Don't pack initially - will be shown when chiral toggle is enabled
        ttk.Label(self.phase_colormap_frame, text="Phase Colormap:").pack(anchor="w")
        self.phase_colormap_var = tk.StringVar(value="bwr")
        diverging_cmaps = ['bwr', 'coolwarm', 'hsv', 'RdBu', 'RdGy', 'PiYG', 'seismic', 'twilight', 'twilight_shifted']
        self.phase_colormap_combo = ttk.Combobox(self.phase_colormap_frame, textvariable=self.phase_colormap_var,
                                             values=diverging_cmaps, state="readonly")
        self.phase_colormap_combo.pack(fill="x", pady=(0,5))
        ToolTip(self.phase_colormap_combo, text="Colormap for chirality plot (diverging/circular preferred).")
        self.phase_colormap_combo.bind("<<ComboboxSelected>>", lambda event: self._on_phase_colormap_change())

        # Global Intensity Scaling (only for K-Grid)
        self.global_scale_frame = ttk.Frame(plot_content_frame)
        # Don't pack initially - will be shown by _update_viz_controls_state
        self.kgrid_global_scale_var = tk.BooleanVar(value=False)
        global_scale_check = ttk.Checkbutton(self.global_scale_frame, text="Global Intensity Scaling", 
                                           variable=self.kgrid_global_scale_var, 
                                           command=lambda: self._plot_kgrid_heatmap(self.kgrid_freq_slider.get()) if hasattr(self, 'kgrid_freq_slider') else None)
        global_scale_check.pack(anchor="w", pady=(0,10))
        
        # Generate plot button
        self.plot_button = ttk.Button(plot_content_frame, text="Generate Plot", 
                                     command=self._generate_plot, state="disabled")
        self.plot_button.pack(fill="x", pady=(10,0))
        
        # Plot status
        self.plot_status_var = tk.StringVar(value="No plot generated")
        ttk.Label(plot_content_frame, textvariable=self.plot_status_var, foreground="blue").pack(anchor="w", pady=(5,0))

        # Save functionality section (organized by function) - compressed spacing
        save_section = ttk.Frame(plot_content_frame)
        save_section.pack(fill="x", pady=(8,0))
        self.save_section = save_section  # Store reference for positioning
        
        ttk.Label(save_section, text="Save Options:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0,3))
        
        # ===== SAVE DATA SECTION ===== - compressed spacing
        data_save_frame = ttk.LabelFrame(save_section, text="Save Data", padding=(6, 2))
        data_save_frame.pack(fill="x", pady=(0,3))
        
        # Save data button
        self.save_data_button = ttk.Button(data_save_frame, text="Save Plot Data", 
                                          command=self._save_plot_data, state="disabled")
        self.save_data_button.pack(fill="x", pady=(0,3))
        
        # Data options frame
        data_options_frame = ttk.Frame(data_save_frame)
        data_options_frame.pack(fill="x")
        
        # Custom filename
        ttk.Label(data_options_frame, text="Filename (optional):").pack(anchor="w")
        self.custom_data_filename_var = tk.StringVar()
        data_name_entry = ttk.Entry(data_options_frame, textvariable=self.custom_data_filename_var)
        data_name_entry.pack(fill="x", pady=(1,3))
        
        # Data format
        format_frame = ttk.Frame(data_options_frame)
        format_frame.pack(fill="x")
        ttk.Label(format_frame, text="Format:").pack(side="left")
        self.save_format_var = tk.StringVar(value="npy")
        format_combo = ttk.Combobox(format_frame, textvariable=self.save_format_var,
                                   values=["npy", "csv"], state="readonly", width=8)
        format_combo.pack(side="left", padx=(5,0))
        ToolTip(format_combo, text="File format for saving plot data (.npy for NumPy arrays, .csv for comprehensive CSV)")
        
        # ===== SAVE PLOT SECTION ===== - compressed spacing
        plot_save_frame = ttk.LabelFrame(save_section, text="Save Plot Image", padding=(6, 2))
        plot_save_frame.pack(fill="x", pady=(0,3))
        
        # Save plot button
        self.save_plot_image_button = ttk.Button(plot_save_frame, text="Save Current Plot", 
                                                command=self._save_current_plot, state="disabled")
        self.save_plot_image_button.pack(fill="x", pady=(0,3))
        
        # Plot options frame
        plot_options_frame = ttk.Frame(plot_save_frame)
        plot_options_frame.pack(fill="x")
        
        # Custom filename
        ttk.Label(plot_options_frame, text="Filename (optional):").pack(anchor="w")
        self.custom_plot_filename_var = tk.StringVar()
        plot_name_entry = ttk.Entry(plot_options_frame, textvariable=self.custom_plot_filename_var)
        plot_name_entry.pack(fill="x", pady=(1,3))
        
        # Format and quality options - compressed spacing
        plot_qual_frame = ttk.Frame(plot_options_frame)
        plot_qual_frame.pack(fill="x", pady=(0,3))
        
        # Image format
        ttk.Label(plot_qual_frame, text="Format:").pack(side="left")
        self.image_format_var = tk.StringVar(value="png")
        image_format_combo = ttk.Combobox(plot_qual_frame, textvariable=self.image_format_var,
                                         values=["png", "jpg", "svg", "pdf"], state="readonly", width=6)
        image_format_combo.pack(side="left", padx=(5,10))
        ToolTip(image_format_combo, text="Image format (PNG/JPG for raster, SVG/PDF for vector)")
        
        # DPI control
        ttk.Label(plot_qual_frame, text="DPI:").pack(side="left")
        self.plot_dpi_var = tk.IntVar(value=300)
        dpi_combo = ttk.Combobox(plot_qual_frame, textvariable=self.plot_dpi_var,
                                values=[150, 300, 600, 1200], state="readonly", width=6)
        dpi_combo.pack(side="left", padx=(5,10))
        ToolTip(dpi_combo, text="Resolution for raster images (150=screen, 300=print, 600=high quality)")
        
       
        # Aspect ratio frame - compressed spacing
        aspect_frame = ttk.Frame(plot_options_frame)
        aspect_frame.pack(fill="x")
        
        ttk.Label(aspect_frame, text="Aspect Ratio (optional):").pack(anchor="w")
        self.aspect_ratio_var = tk.StringVar()
        aspect_entry = ttk.Entry(aspect_frame, textvariable=self.aspect_ratio_var, width=20)
        aspect_entry.pack(anchor="w", pady=(1,0))
        ToolTip(aspect_entry, text="Custom aspect ratio (e.g., 'equal', '1:1', '4:3', '16:9') or leave empty for auto")
        
        # ===== SAVE GIF SECTION ===== - compressed spacing
        gif_save_frame = ttk.LabelFrame(save_section, text="Save K-Grid Animation", padding=(6, 2))
        gif_save_frame.pack(fill="x", pady=(0,3))
        
        # Save gif button
        self.save_kgrid_gif_button = ttk.Button(gif_save_frame, text="Save K-Grid Animation", 
                                               command=self._save_kgrid_gif, state="disabled")
        self.save_kgrid_gif_button.pack(fill="x", pady=(0,3))
        
        # GIF options frame
        gif_options_frame = ttk.Frame(gif_save_frame)
        gif_options_frame.pack(fill="x")
        
        # Custom filename
        ttk.Label(gif_options_frame, text="Filename (optional):").pack(anchor="w")
        self.custom_animation_filename_var = tk.StringVar()
        anim_name_entry = ttk.Entry(gif_options_frame, textvariable=self.custom_animation_filename_var)
        anim_name_entry.pack(fill="x", pady=(1,3))
        
        # Frame rate - compressed spacing
        fps_frame = ttk.Frame(gif_options_frame)
        fps_frame.pack(fill="x")
        
        ttk.Label(fps_frame, text="Frame Rate (fps):").pack(side="left")
        self.gif_fps_var = tk.DoubleVar(value=5.0)
        fps_entry = ttk.Entry(fps_frame, textvariable=self.gif_fps_var, width=10)
        fps_entry.pack(side="left", padx=(5,0))
        ToolTip(fps_entry, text="Animation frame rate in frames per second (e.g., 5.0, 10.0, 30.0)")

        # Spacer to push everything up
        self.viz_spacer_frame = ttk.Frame(plot_content_frame)
        self.viz_spacer_frame.pack(fill="both", expand=True)

    def _create_ised_tab(self, notebook):
        """Create iSED reconstruction parameters tab"""
        ised_frame = ttk.Frame(notebook)
        notebook.add(ised_frame, text="Reconstruction")
        
        ttk.Label(ised_frame, text="Click Parameters:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        # Selected point display
        self.selected_k_var = tk.StringVar(value="k = Not selected")
        self.selected_w_var = tk.StringVar(value="ω = Not selected")
        
        ttk.Label(ised_frame, textvariable=self.selected_k_var, foreground="red").pack(anchor="w")
        ttk.Label(ised_frame, textvariable=self.selected_w_var, foreground="red").pack(anchor="w", pady=(0,10))
        
        # Reconstruction parameters
        ttk.Label(ised_frame, text="Reconstruction Parameters:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        # Number of frames and rescaling factor
        self.n_frames_var = tk.IntVar(value=100)
        self._create_labeled_entry(ised_frame, "Animation Frames:", self.n_frames_var)
        
        self.rescale_var = tk.StringVar(value="auto")
        _, rescale_entry = self._create_labeled_entry(ised_frame, "Rescaling Factor:", self.rescale_var, 
                                                     pack_kwargs={'pady': (0,10)})
        ToolTip(rescale_entry, text="Rescaling factor for iSED atomic displacements.\n'auto' for automatic scaling, or a numerical value (e.g., 0.1, 1.0, 10.0).")
        
        # 3D Visualization Controls
        ttk.Label(ised_frame, text="3D Visualization Controls:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(20,5))
        
        # Atom sizes using helper method
        self.atom_size_type1_var = tk.DoubleVar(value=0.3)
        self._create_labeled_scale(ised_frame, "Atom Type 1 Size:", self.atom_size_type1_var, 
                                  0.05, 2.0, self._update_atom_size)
        
        self.atom_size_type2_var = tk.DoubleVar(value=0.3)
        self._create_labeled_scale(ised_frame, "Atom Type 2 Size:", self.atom_size_type2_var, 
                                  0.05, 2.0, self._update_atom_size)
        
        # Atom transparency using helper method
        self.atom_alpha_var = tk.DoubleVar(value=0.8)
        self._create_labeled_scale(ised_frame, "Transparency:", self.atom_alpha_var, 
                                  0.1, 1.0, self._update_atom_alpha, "{:.1f}")
        
        # Extra padding before reconstruct button
        ttk.Frame(ised_frame).pack(pady=(5,0))
        
        # Reconstruct button
        self.ised_button = ttk.Button(ised_frame, text="Reconstruct iSED", 
                                     command=self._reconstruct_ised, state="disabled")
        self.ised_button.pack(fill="x", pady=(20,10))
        
        # iSED Saving Section
        ttk.Label(ised_frame, text="Save Trajectory:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        # Custom iSED trajectory filename
        ised_save_frame = ttk.Frame(ised_frame)
        ised_save_frame.pack(fill="x", pady=(0,5))
        
        ttk.Label(ised_save_frame, text="Filename (optional):").pack(anchor="w")
        self.custom_ised_filename_var = tk.StringVar()
        ised_name_entry = ttk.Entry(ised_save_frame, textvariable=self.custom_ised_filename_var)
        ised_name_entry.pack(fill="x", pady=(2,0))
        ToolTip(ised_name_entry, text="Custom name for saved iSED trajectory (leave empty for auto-generated name)")
        
        # Save iSED trajectory button
        self.save_ised_button = ttk.Button(ised_frame, text="Save iSED Trajectory", 
                                          command=self._save_ised_trajectory, state="disabled")
        self.save_ised_button.pack(fill="x", pady=(5,5))
        
        ttk.Label(ised_frame, text="Saved files will use the output directory from I/O tab", 
                 font=("Arial", 8), foreground="gray").pack(anchor="w")
        
        # Animation speed
        self.anim_fps_var = tk.DoubleVar(value=10.0)
        self._create_labeled_scale(ised_frame, "Animation Speed (fps):", self.anim_fps_var, 
                                  1.0, 30.0, self._update_anim_speed, "{:.1f}")
        
        # Animation controls
        anim_control_frame = ttk.Frame(ised_frame)
        anim_control_frame.pack(fill="x", pady=(10,5))
        
        self.play_pause_button = ttk.Button(anim_control_frame, text="⏸️ Pause", 
                                           command=self._toggle_animation)
        self.play_pause_button.pack(side="left", padx=(0,5))
        
        ttk.Button(anim_control_frame, text="🔄 Reset View", 
                  command=self._reset_3d_view).pack(side="left", padx=(0,5))
        
        # View motion button
        self.view_motion_button = ttk.Button(ised_frame, text="View in External Viewer", 
                                           command=self._view_atomic_motion, state="disabled")
        self.view_motion_button.pack(fill="x", pady=(10,5))

        # iSED status
        self.ised_status_var = tk.StringVar(value="No iSED reconstruction")
        ttk.Label(ised_frame, textvariable=self.ised_status_var, foreground="blue").pack(anchor="w")

    def _create_plot_panel(self, parent):
        """Create the right panel for plots"""
        plot_frame = ttk.Frame(parent)
        parent.add(plot_frame)
        
        # Create notebook for multiple plots
        self.plot_notebook = ttk.Notebook(plot_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # SED Plot Tab
        self._create_sed_plot_tab()
        
        # iSED Plot Tab  
        self._create_ised_plot_tab()

    def _create_sed_plot_tab(self):
        """Create SED plot display tab"""
        self.sed_plot_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.sed_plot_frame, text="Reciprocal Space")
        
        # Create matplotlib figure for SED with better size and layout
        self.sed_fig = Figure(figsize=(8, 8), dpi=100)  # Adjusted for a more square figure, can be tuned
        self.sed_ax = self.sed_fig.add_subplot(111)
        # Title and labels are set during plot generation for dynamic content (chiral vs non-chiral)
        
        # Set aspect for the axes box to be equal, making the plot area square
        # This will be applied more robustly in _generate_sed_plot after data ranges are known
        # self.sed_ax.set_aspect('equal', adjustable='box') 
        
        # Create canvas
        self.sed_canvas = FigureCanvasTkAgg(self.sed_fig, self.sed_plot_frame)
        self.sed_canvas.draw() # Initial draw
        self.sed_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.sed_plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.sed_toolbar = NavigationToolbar2Tk(self.sed_canvas, toolbar_frame)
        self.sed_toolbar.update()
        
    def _create_ised_plot_tab(self):
        """Create iSED visualization tab"""
        self.ised_plot_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.ised_plot_frame, text="Real Space")
        
        # Full width atomic motion visualization
        header_frame = ttk.Frame(self.ised_plot_frame)
        header_frame.pack(fill="x", pady=(5,5))
        
        ttk.Label(header_frame, text="3D Atomic Motion Visualization", 
                 font=("Arial", 12, "bold")).pack(side="left")
        
        # Navigation help
        ttk.Label(header_frame, text="Use toolbar below to zoom, pan, and rotate", 
                 font=("Arial", 9), foreground="gray").pack(side="right")
        
        # Create matplotlib figure for 3D motion
        self.motion_fig = Figure(figsize=(14, 10), dpi=100)
        self.motion_ax = self.motion_fig.add_subplot(111, projection='3d')
        
        # Enable mouse interaction for 3D plots and set up navigation
        self.motion_ax.mouse_init()
        
        # Configure 3D view for better interaction
        self.motion_ax.view_init(elev=20, azim=45)  # Set initial view angle
        
        self.motion_canvas = FigureCanvasTkAgg(self.motion_fig, self.ised_plot_frame)
        self.motion_canvas.draw()
        self.motion_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Enable navigation explicitly
        self.motion_canvas.get_tk_widget().configure(cursor="crosshair")
        
        # Add toolbar for 3D navigation with zoom capabilities
        toolbar_frame = ttk.Frame(self.ised_plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.motion_toolbar = NavigationToolbar2Tk(self.motion_canvas, toolbar_frame)
        self.motion_toolbar.update()
        
        # Force toolbar to be visible and configure for 3D
        self.motion_toolbar.pack(fill=tk.X)
        
        # Add custom 3D navigation hint since some toolbar buttons might not work optimally with 3D
        nav_hint_frame = ttk.Frame(toolbar_frame)
        nav_hint_frame.pack(fill=tk.X, pady=(2,0))
        ttk.Label(nav_hint_frame, text="3D Navigation: Left-click+drag=rotate, Right-click+drag=zoom, Middle-click+drag=pan", 
                 font=("Arial", 8), foreground="gray").pack()
        
    def _setup_event_handlers(self):
        """Setup event handlers for interactive features"""
        # Connect click event to SED plot
        self.sed_canvas.mpl_connect('button_press_event', self._on_sed_plot_click)
        
    def _browse_trajectory(self):
        """Browse for trajectory file"""
        filename = filedialog.askopenfilename(
            title="Select Trajectory File",
            filetypes=[
                ("LAMMPS Trajectory", "*.lammpstrj"),
                ("XYZ Files", "*.xyz"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.trajectory_var.set(filename)
            self.trajectory_file = filename
            self.load_button.config(state="normal")
            
    def _load_trajectory(self):
        """Load the selected trajectory file"""
        if not self.trajectory_file:
            messagebox.showerror("Error", "Please select a trajectory file first")
            return
            
        # Create progress dialog
        progress = ProgressDialog(self.root, "Loading Trajectory")
        progress.update_message("Checking for cached files...", f"File: {Path(self.trajectory_file).name}")
        
        try:
            # Load trajectory on main thread to avoid OVITO threading issues
            loader = TrajectoryLoader(
                filename=self.trajectory_file,
                dt=self.dt_var.get(),
                file_format=self.format_var.get()
            )
            
            # Check if we need OVITO loading
            cache_stem = Path(self.trajectory_file).parent / Path(self.trajectory_file).stem
            npy_files = {
                'positions': cache_stem.with_suffix('.positions.npy'),
                'velocities': cache_stem.with_suffix('.velocities.npy'),
                'types': cache_stem.with_suffix('.types.npy'),
                'box_matrix': cache_stem.with_suffix('.box_matrix.npy')
            }
            
            if all(f.exists() for f in npy_files.values()):
                progress.update_message("Loading from cached .npy files...", "This should be fast!")
            else:
                progress.update_message("Loading trajectory file...", 
                                      "Processing with OVITO (first time may take a while).\nFuture loads will be much faster using cached files.")
            
            # Process the loading with periodic GUI updates
            trajectory = loader.load()
            
            progress.update_message("Initializing SED calculator...", "Almost done...")
            
            # Initialize SED calculator
            self.sed_calculator = SEDCalculator(
                traj=trajectory,
                nx=self.nx_var.get(),
                ny=self.ny_var.get(),
                nz=self.nz_var.get()
            )
            
            # Update status
            self.status_var.set(f"Trajectory loaded: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
            self.calc_sed_button.config(state="normal")
            
        except Exception as e:
            logger.error(f"Error loading trajectory: {e}")
            error_msg = str(e)
            if "subprocess failed" in error_msg.lower() or "ovito" in error_msg.lower():
                error_msg = f"Failed to load trajectory file.\n\nError: {str(e)}\n\nTip: Try running the basic_sed_analysis.py script first to check if OVITO works in your environment."
            else:
                error_msg = f"Failed to load trajectory:\n{str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error loading trajectory")
        
        finally:
            # Close progress dialog
            progress.close()
        
    def _calculate_sed(self):
        """Calculate SED dispersion or k-grid based on mode"""
        if not self.sed_calculator:
            messagebox.showerror("Error", "Please load a trajectory first")
            return
        
        calc_mode = self.calc_mode_var.get()
        
        if calc_mode == "K-Path":
            self._calculate_kpath_sed()
        else:  # K-Grid
            self._calculate_kgrid_sed()

    def _calculate_kpath_sed(self):
        """Calculate K-Path SED dispersion"""
        def calc_worker():
            try:
                # Thread-safe GUI updates
                self.root.after(0, lambda: self.sed_status_var.set("Calculating K-Path SED..."))
                
                # Parse direction (existing logic)
                direction_str = self.direction_var.get().strip()
                try:
                    if direction_str.startswith('[') and direction_str.endswith(']'):
                        direction = ast.literal_eval(direction_str)
                        if not (isinstance(direction, list) and len(direction) == 3 and all(isinstance(x, (int, float)) for x in direction)):
                            logger.warning(f"Invalid format for direction list: {direction_str}. Using default.")
                            direction = [1, 1, 0]
                    elif direction_str.lower() in ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']:
                        direction = direction_str.lower()
                    else:
                        logger.warning(f"Invalid direction string: {direction_str}. Using default.")
                        direction = [1, 1, 0]
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Error parsing direction string '{direction_str}': {e}. Using default.")
                    direction = [1, 1, 0]
                
                basis_types = self._get_basis_atom_types()
                
                k_mags, k_vecs = self.sed_calculator.get_k_path(
                    direction_spec=direction,
                    bz_coverage=self.bz_coverage_var.get(),
                    n_k=self.n_k_var.get()
                )
                
                # Rest of existing k-path calculation logic...
                summation_mode_to_use = self.summation_mode_var.get()
                if self.chiral_sed_var.get():
                    if summation_mode_to_use != 'coherent':
                        logger.info("Chirality calculation selected, forcing coherent summation mode.")
                        summation_mode_to_use = 'coherent'

                calc_kwargs = {
                    'k_points_mags': k_mags,
                    'k_vectors_3d': k_vecs,
                    'basis_atom_types': basis_types,
                    'summation_mode': summation_mode_to_use
                }
                
                sed_object = self.sed_calculator.calculate(**calc_kwargs)
                logger.info(f"Base SED calculation complete with {summation_mode_to_use} mode.")

                calculated_phase = None
                if self.chiral_sed_var.get():
                    if hasattr(sed_object, 'sed') and sed_object.sed is not None and sed_object.is_complex:
                        if sed_object.sed.ndim == 3 and sed_object.sed.shape[-1] >= 2:
                            axis = self.chiral_axis_var.get()
                            if axis == 'x':
                                idx1, idx2 = 1, 2
                            elif axis == 'y':
                                idx1, idx2 = 0, 2
                            else:
                                idx1, idx2 = 0, 1
                            Z1 = sed_object.sed[:,:,idx1]
                            Z2 = sed_object.sed[:,:,idx2]
                            logger.info(f"Calculating chiral phase using axis {axis} (components {idx1} and {idx2}).")
                            calculated_phase = self.sed_calculator.calculate_chiral_phase(
                                Z1=Z1, Z2=Z2, angle_range_opt="C"
                            )
                            logger.info("Chiral phase calculation complete.")
                
                self.sed_result = SED(
                    sed=sed_object.sed,
                    freqs=sed_object.freqs,
                    k_points=sed_object.k_points,
                    k_vectors=sed_object.k_vectors,
                    k_grid_shape=sed_object.k_grid_shape,
                    phase=calculated_phase, # This will be None if not calculated
                    is_complex=sed_object.is_complex
                )
                
                # Store calculation type
                self.sed_calculation_type = "K-Path"
                
                # Update visualization controls based on available data
                self._update_viz_controls_state() # Call to update controls

                self.root.after(0, lambda: self.sed_status_var.set("SED calculation complete"))
                self.root.after(0, lambda: self.plot_button.config(state="normal"))
                
            except Exception as e:
                logger.error(f"Error calculating SED: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to calculate SED:\n{str(e)}"))
                self.root.after(0, lambda: self.sed_status_var.set("Error calculating SED"))
                
        thread = threading.Thread(target=calc_worker)
        thread.daemon = True
        thread.start()
        
    def _generate_sed_plot(self):
        """Generate and display SED plot"""
        if not self.sed_result:
            messagebox.showerror("Error", "Please calculate SED first")
            return
            
        try:
            self.plot_status_var.set("Generating plot...")
            
            # Clear previous plot completely
            self.sed_ax.clear()
            
            # Remove previous colorbar if it exists
            self._cleanup_colorbar()
                    
            # Remove click marker if it exists
            if self.click_marker is not None:
                try:
                    self.click_marker.remove()
                except Exception as e:
                    logger.debug(f"Error removing click marker: {e}")
                self.click_marker = None
            
            # Calculate intensity
            # Check if phase information is present (indicating chiral calculation was done)
            is_chiral_plot = hasattr(self.sed_result, 'phase') and self.sed_result.phase is not None

            if not hasattr(self.sed_result, 'sed') or self.sed_result.sed is None:
                messagebox.showerror("Plot Error", "SED result data (sed attribute) not found or is invalid.")
                self.plot_status_var.set("Error: SED data missing for plot")
                return

            # Apply frequency filtering
            positive_freq_mask = self.sed_result.freqs >= 0
            plot_freqs = self.sed_result.freqs[positive_freq_mask]
            
            # Determine data to plot and colorbar label
            colorbar_label = "Intensity" # Default
            data_to_plot = None

            # Use chiral toggle instead of dropdown
            plot_chiral = self.plot_chiral_var.get()
            # Determine effective_cmap based on chiral toggle
            if plot_chiral:
                effective_cmap = self.phase_colormap_var.get()
            else: # SED mode
                effective_cmap = self.colormap_var.get()

            if plot_chiral:
                if is_chiral_plot and hasattr(self.sed_result, 'phase') and self.sed_result.phase is not None:
                    data_to_plot = self.sed_result.phase[positive_freq_mask, :]
                    colorbar_label = "Phase Angle (radians)"
                    plot_title = "Chirality - Click to select (k,ω) point"
                    logger.info("Plotting Chirality.")
                else:
                    messagebox.showerror("Plot Error", "Chirality plot selected, but phase data is not available.")
                    self.plot_status_var.set("Error: Chirality data unavailable.")
                    self.plot_chiral_var.set(False)  # Reset toggle
                    plot_chiral = False
            if not plot_chiral:  # SED mode
                if is_chiral_plot:
                    if not self.sed_result.is_complex:
                        messagebox.showerror("Plot Error", "Chirality plot selected, but SED data is not complex.")
                        self.plot_status_var.set("Error: Chirality plot needs complex SED data.")
                        return
                    intensity_values = np.sum(np.abs(self.sed_result.sed)**2, axis=-1)
                    data_to_plot = intensity_values[positive_freq_mask]
                    colorbar_label = "Intensity"
                    plot_title = "SED - Click to select (k,ω) point"
                    logger.info("Plotting SED (derived from coherent complex data).")
                else:
                    if self.sed_result.is_complex:
                        intensity_values = np.sum(np.abs(self.sed_result.sed)**2, axis=-1)
                        data_to_plot = intensity_values[positive_freq_mask]
                        logger.info("Plotting coherent SED.")
                    else:
                        if self.sed_result.sed.ndim == 1:
                            logger.warning("Incoherent SED data is 1D, attempting to reshape or skip. This might indicate an issue.")
                        data_to_plot = self.sed_result.sed[positive_freq_mask, :]
                        logger.info("Plotting incoherent SED.")
                    colorbar_label = "Intensity"
                    plot_title = "SED - Click to select (k,ω) point"
                
                # Apply intensity scaling only for SED (not for chirality/phase)
                data_to_plot = self._apply_intensity_scaling(data_to_plot, self.intensity_scale_var.get())

            if data_to_plot is None:
                messagebox.showerror("Plot Error", "No data available to plot. Please check settings and recalculate SED.")
                self.plot_status_var.set("Error: No data to plot")
                return

            # Apply max frequency filter if specified (to the chosen data_to_plot and plot_freqs)
            if self.max_freq_var.get().strip():
                try:
                    max_freq_val = float(self.max_freq_var.get())
                    freq_mask_for_max = plot_freqs <= max_freq_val
                    plot_freqs = plot_freqs[freq_mask_for_max]
                    data_to_plot = data_to_plot[freq_mask_for_max]
                except ValueError:
                    pass

            # Ensure data_to_plot is 2D
            if data_to_plot.ndim != 2:
                messagebox.showerror("Plot Error", f"Data for plotting is not 2D (got shape {data_to_plot.shape}). Cannot generate plot.")
                self.plot_status_var.set("Error: Data not 2D for plot")
                return

            # Create meshgrid for plotting
            k_plot_points = self.sed_result.k_points
            if k_plot_points.ndim > 1:
                k_plot_points = k_plot_points.flatten()
            if k_plot_points.shape[0] != data_to_plot.shape[1]:
                messagebox.showerror("Plot Error", f"Shape mismatch: k_points ({k_plot_points.shape[0]}) and data_to_plot columns ({data_to_plot.shape[1]}). Cannot generate plot.")
                self.plot_status_var.set("Error: Plot data shape mismatch")
                return
            K, F = np.meshgrid(k_plot_points, plot_freqs)

            # Ensure data_to_plot has finite values
            if not np.any(np.isfinite(data_to_plot)):
                messagebox.showerror("Plot Error", "Data for plotting contains no finite values (e.g., all NaN or Inf). Cannot generate plot.")
                self.plot_status_var.set("Error: Non-finite data for plot")
                return

            # Remove any existing colorbar and its axes before plotting
            self._cleanup_colorbar()

            # Plot
            im = self.sed_ax.pcolormesh(K, F, data_to_plot, 
                                       cmap=effective_cmap, 
                                       shading='auto')

            # Set proper aspect ratio and limits
            k_min, k_max = k_plot_points.min(), k_plot_points.max()
            f_min, f_max = plot_freqs.min(), plot_freqs.max()
            self.sed_ax.set_xlim(k_min, k_max)
            self.sed_ax.set_ylim(f_min, f_max)
            self.sed_ax.set_xlabel("k (2π/Å)", fontsize=12)
            self.sed_ax.set_ylabel("Frequency (THz)", fontsize=12)
            self.sed_ax.set_title(plot_title, fontsize=14)

            # Always add colorbar after plotting
            try:
                divider = make_axes_locatable(self.sed_ax)
                self.sed_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
                self.sed_colorbar_ax.clear()
                self.sed_colorbar = self.sed_fig.colorbar(im, cax=self.sed_colorbar_ax)
                self.sed_colorbar.set_label(colorbar_label, fontsize=12)
            except Exception as e:
                logger.error(f"Error creating/updating colorbar: {e}", exc_info=True)
                self.sed_colorbar = None
                self.sed_colorbar_ax = None
            
            # Store plot data for click detection
            self.current_plot_data = {
                'k_points': k_plot_points, # Use potentially flattened k_points
                'freqs': plot_freqs,
                'K': K,
                'F': F
            }
            
            # Redraw canvas with better layout management
            try:
                # Use tight_layout, it should respect the colorbar axes created by make_axes_locatable
                self.sed_fig.tight_layout()
                self.sed_canvas.draw()
            except Exception as e:
                logger.error(f"Error drawing canvas with tight_layout: {e}")
                try:
                    self.sed_canvas.draw() # Fallback
                except Exception as ed:
                    logger.error(f"Fallback draw failed: {ed}")
                
            self.plot_status_var.set("SED plot generated - click on plot to select point")
            
        except Exception as e:
            logger.error(f"Error generating plot: {e}")
            messagebox.showerror("Error", f"Failed to generate plot:\n{str(e)}")
            self.plot_status_var.set("Error generating plot")
            
    def _on_sed_plot_click(self, event):
        """Handle clicks on SED plot"""
        if event.inaxes != self.sed_ax or not self.current_plot_data:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        try:
            # Find closest point
            k_click = event.xdata
            w_click = event.ydata
            
            k_points = self.current_plot_data['k_points']
            freqs = self.current_plot_data['freqs']
            
            # Find closest k and frequency indices
            k_idx = np.argmin(np.abs(k_points - k_click))
            f_idx = np.argmin(np.abs(freqs - w_click))
            
            # Get actual values
            k_actual = k_points[k_idx]
            w_actual = freqs[f_idx]
            
            # Remove previous marker if it exists
            if self.click_marker is not None:
                try:
                    self.click_marker.remove()
                except Exception as e:
                    logger.debug(f"Error removing previous marker: {e}")
            
            # Add green plus marker at clicked location
            try:
                self.click_marker = self.sed_ax.plot(k_actual, w_actual, marker='+', 
                                                   color='green', markersize=15, 
                                                   markeredgewidth=3)[0]
                # Redraw canvas to show marker
                self.sed_canvas.draw()
            except Exception as e:
                logger.error(f"Error adding click marker: {e}")
                self.click_marker = None
            
            # Update display
            self.selected_k_var.set(f"k = {k_actual:.3f} (2π/Å)")
            self.selected_w_var.set(f"ω = {w_actual:.3f} THz")
            
            # Store selected values
            self.selected_k = k_actual
            self.selected_w = w_actual
            
            # Enable iSED reconstruction
            self.ised_button.config(state="normal")
            
            logger.info(f"Selected point: k={k_actual:.3f}, ω={w_actual:.3f}")
            
        except Exception as e:
            logger.error(f"Error handling plot click: {e}")
            
    def _browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get()
        )
        if directory:
            self.output_dir_var.set(directory)
            
    def _reconstruct_ised(self):
        """Perform iSED reconstruction for selected point"""
        if not hasattr(self, 'selected_k') or not hasattr(self, 'selected_w'):
            messagebox.showerror("Error", "Please select a point on the SED plot first")
            return
            
        # Disable button during reconstruction
        self.ised_button.config(state="disabled")
        self.ised_status_var.set("Reconstructing iSED...")
        
        # Stop any running animation
        self.animation_running = False
        if hasattr(self, 'current_animation_id') and self.current_animation_id:
            self.root.after_cancel(self.current_animation_id)
            self.current_animation_id = None
        
        # Run iSED reconstruction in main thread to avoid matplotlib threading issues
        try:
            # Always use temporary directory for initial reconstruction
            temp_dir_obj = tempfile.TemporaryDirectory()
            self.current_temp_ised_dir = Path(temp_dir_obj.name)
            dump_file = self.current_temp_ised_dir / "ised_motion.dump"
            output_dir = self.current_temp_ised_dir
            # Store the TemporaryDirectory object itself to prevent premature cleanup
            self._current_temp_dir_obj = temp_dir_obj
            
            # Parse direction for iSED
            direction_str = self.direction_var.get().strip()
            k_dir_ised = [1,1,0] # Default
            try:
                if direction_str.startswith('[') and direction_str.endswith(']'):
                    parsed_direction = ast.literal_eval(direction_str)
                    if isinstance(parsed_direction, list) and len(parsed_direction) == 3 and all(isinstance(x, (int, float)) for x in parsed_direction):
                        k_dir_ised = parsed_direction # Pass the list directly
                    else:
                        logger.warning(f"Invalid format for iSED direction list: {direction_str}. Using default [1,1,0].")
                elif direction_str.lower() in ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']:
                    k_dir_ised = direction_str.lower()
                elif direction_str: # Non-empty, but not recognized
                    logger.warning(f"Unrecognized iSED direction string: {direction_str}. Using default [1,1,0].")
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Error parsing iSED direction string '{direction_str}': {e}. Using default [1,1,0].")
            
            # Always use auto-detected lattice parameter
            lat_param = np.linalg.norm(self.sed_calculator.a1)
            
            # Get basis types for iSED
            basis_types_ised = None
            if self.basis_types_var.get().strip():
                try:
                    basis_types_ised = [int(x.strip()) for x in self.basis_types_var.get().split(',')]
                except ValueError:
                    pass
            
            # Process in chunks to keep GUI responsive
            def process_ised():
                try:
                    # Perform iSED reconstruction (NO plot_dir_ised to avoid threading issues)
                    self.sed_calculator.ised(
                        k_dir_spec=k_dir_ised,
                        k_target=self.selected_k,
                        w_target=self.selected_w,
                        char_len_k_path=lat_param,
                        nk_on_path=self.n_k_var.get(),
                        bz_cov_ised=self.bz_coverage_var.get(),
                        basis_atom_types_ised=basis_types_ised,
                        rescale_factor=self.rescale_var.get(),
                        n_recon_frames=self.n_frames_var.get(),
                        dump_filepath=str(dump_file),
                        plot_dir_ised=None  # Disable auto-plotting to avoid threading issues
                    )
                    
                    self.ised_result_path = str(dump_file)
                    self.ised_plot_dir = output_dir
                    
                    # Load and display atomic motion
                    self._load_atomic_motion()
                    
                    self.ised_status_var.set("iSED reconstruction complete (temporary)")
                        
                    self.view_motion_button.config(state="normal")
                    self.ised_button.config(state="normal")
                    
                    # Update visualization controls to enable save button
                    self._update_viz_controls_state()
                    
                    # Switch to Real Space tab
                    self.plot_notebook.select(1)
                    
                except Exception as e:
                    error_msg = f"Error in iSED reconstruction: {e}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    messagebox.showerror("Error", f"Failed to reconstruct iSED:\n{str(e)}")
                    self.ised_status_var.set("Error in iSED reconstruction")
                    self.ised_button.config(state="normal")
            
            # Schedule the processing
            self.root.after(10, process_ised)
            
        except Exception as e:
            logger.error(f"Error setting up iSED reconstruction: {e}")
            messagebox.showerror("Error", f"Failed to setup iSED reconstruction:\n{str(e)}")
            self.ised_status_var.set("Error in iSED reconstruction")
            self.ised_button.config(state="normal")
            
    def _load_atomic_motion(self):
        """Load and display atomic motion visualization"""
        if not self.ised_result_path or not Path(self.ised_result_path).exists():
            logger.error(f"iSED result path doesn't exist: {self.ised_result_path}")
            return
            
        try:
            logger.info(f"Loading atomic motion from: {self.ised_result_path}")
            
            # Read the LAMMPS dump file
            positions_frames = self._read_lammps_dump(self.ised_result_path)
            
            logger.info(f"Read {len(positions_frames)} frames from dump file")
            
            if positions_frames:
                # Display the first frame and start animation
                self._animate_atomic_motion(positions_frames)
                logger.info("Atomic motion animation started")
            else:
                logger.error("No frames read from dump file")
                
        except Exception as e:
            logger.error(f"Error loading atomic motion: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _read_lammps_dump(self, dump_file: str) -> list:
        """Read positions from LAMMPS dump file"""
        try:
            logger.info(f"Reading LAMMPS dump file: {dump_file}")
            frames = []
            current_frame = []
            
            with open(dump_file, 'r') as f:
                lines = f.readlines()
                
            logger.info(f"Read {len(lines)} lines from dump file")
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith("ITEM: TIMESTEP"):
                    if current_frame:
                        frames.append(np.array(current_frame))
                        current_frame = []
                    i += 1
                    continue
                    
                if line.startswith("ITEM: NUMBER OF ATOMS"):
                    n_atoms = int(lines[i + 1].strip())
                    logger.debug(f"Found {n_atoms} atoms in frame")
                    i += 2
                    continue
                    
                if line.startswith("ITEM: ATOMS"):
                    # Read atom data
                    i += 1
                    for j in range(n_atoms):
                        if i + j < len(lines):
                            parts = lines[i + j].strip().split()
                            if len(parts) >= 5:  # id type x y z (5 columns)
                                try:
                                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                                    current_frame.append([x, y, z])
                                except (ValueError, IndexError) as e:
                                    logger.warning(f"Error parsing atom line: {parts}, error: {e}")
                    i += n_atoms
                    continue
                    
                i += 1
                
            if current_frame:
                frames.append(np.array(current_frame))
                
            logger.info(f"Successfully parsed {len(frames)} frames")
            if frames:
                logger.info(f"First frame has {len(frames[0])} atoms")
                
            return frames
            
        except Exception as e:
            logger.error(f"Error reading LAMMPS dump: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
            
    def _animate_atomic_motion(self, frames: list):
        """Animate atomic motion"""
        if not frames:
            logger.error("No frames to animate")
            return
            
        try:
            logger.info(f"Starting animation with {len(frames)} frames")
            
            # Stop any existing animation
            self.animation_running = False
            
            self.motion_ax.clear()
            
            # Use first frame for initial display
            positions = frames[0]
            
            if len(positions) == 0:
                logger.error("First frame has no atoms")
                return
                
            logger.info(f"First frame has {len(positions)} atoms")
            
            # Get atom types from the dump file - need to read them separately
            atom_types = self._get_atom_types_from_dump()
            
            # Plot atoms with proper coloring
            self._plot_atoms_3d(positions, atom_types)
            
            # Set axes labels and title
            self.motion_ax.set_xlabel('X (Å)', fontsize=12)
            self.motion_ax.set_ylabel('Y (Å)', fontsize=12)
            self.motion_ax.set_zlabel('Z (Å)', fontsize=12)
            
            # Show current FPS in title for user feedback
            current_fps = self.anim_fps_var.get() if hasattr(self, 'anim_fps_var') else 10.0
            self.motion_ax.set_title(f'Atomic Motion Animation (Frame 1/{len(frames)}) - {current_fps:.1f} FPS', fontsize=14)
            
            # Set proper view and limits
            self._set_3d_view_limits(positions)
            
            self.motion_canvas.draw()
            logger.info("Drew initial frame")
            
            # Store frames and types for animation
            self.motion_frames = frames
            self.motion_atom_types = atom_types
            self.current_frame_idx = 0
            
            # Start animation
            self._start_motion_animation()
            
        except Exception as e:
            logger.error(f"Error animating atomic motion: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _get_atom_types_from_dump(self):
        """Extract atom types from the LAMMPS dump file"""
        try:
            if not self.ised_result_path:
                return None
                
            atom_types = []
            n_atoms_current_frame = 0
            reading_atoms_for_types = False

            with open(self.ised_result_path, 'r') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line.startswith("ITEM: NUMBER OF ATOMS"):
                        try:
                            n_atoms_current_frame = int(next(f).strip())
                            if not atom_types: 
                                reading_atoms_for_types = True 
                        except (StopIteration, ValueError) as e:
                            logger.error(f"Error parsing N_ATOMS in dump: {e}")
                            return None
                        continue
                    
                    if reading_atoms_for_types and stripped_line.startswith("ITEM: ATOMS"):
                        for _ in range(n_atoms_current_frame):
                            try:
                                atom_line = next(f).strip()
                                parts = atom_line.split()
                                if len(parts) >= 2:
                                    atom_types.append(int(parts[1]))
                                else:
                                    logger.warning(f"Skipping malformed atom line for types: {atom_line}")
                            except (StopIteration, ValueError, IndexError) as e:
                                logger.error(f"Error parsing atom line for types: {e}")
                                return np.array(atom_types) if atom_types else None
                        reading_atoms_for_types = False
                        logger.info(f"Extracted {len(atom_types)} atom types from first frame.")
                        return np.array(atom_types) 
            
            if atom_types:
                logger.info(f"Extracted {len(atom_types)} atom types (fallback).")
                return np.array(atom_types)
            else:
                logger.warning("Could not extract atom types from dump file.")
                return None

        except Exception as e:
            logger.error(f"Error reading atom types: {e}")
            return None

    def _plot_atoms_3d(self, positions, atom_types):
        """Plot atoms in 3D with proper coloring and sizing"""
        alpha = self.atom_alpha_var.get()
        
        if atom_types is not None:
            # Always color by atom type with individual size controls
            unique_types = np.unique(atom_types)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, atom_type in enumerate(unique_types):
                mask = atom_types == atom_type
                if np.any(mask):
                    color = colors[i % len(colors)]
                    
                    # Get size for this atom type
                    if atom_type == 1:
                        atom_size = self.atom_size_type1_var.get()
                    elif atom_type == 2:
                        atom_size = self.atom_size_type2_var.get()
                    else:
                        atom_size = 0.3  # Default for other types, consistent with initial Type 1/2 size
                    
                    # Convert size to matplotlib scatter size (area in points^2)
                    scatter_size = atom_size * 5  # Much smaller scale factor for tiny atoms
                    
                    self.motion_ax.scatter(positions[mask, 0], positions[mask, 1], positions[mask, 2],
                                         c=color, s=scatter_size, alpha=alpha, 
                                         label=f'Type {atom_type} ({np.sum(mask)} atoms)')
            
            # Add legend if multiple types
            if len(unique_types) > 1:
                self.motion_ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                
        else:
            # Fallback if no atom types available
            atom_size = self.atom_size_type1_var.get()
            scatter_size = atom_size * 5  # Much smaller scale factor
            self.motion_ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                 c='red', s=scatter_size, alpha=alpha)
                                 
    def _set_3d_view_limits(self, positions):
        """Set proper 3D view limits and aspect ratio"""
        if len(positions) == 0:
            return
            
        # Calculate bounds with some padding
        margin = 2.0  # Angstrom
        x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin
        y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin
        z_min, z_max = positions[:, 2].min() - margin, positions[:, 2].max() + margin
        
        self.motion_ax.set_xlim(x_min, x_max)
        self.motion_ax.set_ylim(y_min, y_max)
        self.motion_ax.set_zlim(z_min, z_max)
        
        # Set equal aspect ratio
        self.motion_ax.set_box_aspect([1,1,1])
        
    def _update_atom_size(self, value=None):
        """Update atom size in the 3D plot"""
        if hasattr(self, 'motion_frames') and self.motion_frames:
            self._refresh_3d_plot()
            
    def _update_atom_alpha(self, value=None):
        """Update atom transparency in the 3D plot"""
        if hasattr(self, 'motion_frames') and self.motion_frames:
            self._refresh_3d_plot()
            
    def _update_anim_speed(self, value=None):
        """Update animation speed - this will take effect on the next frame"""
        # The new speed will be picked up in the next animation frame automatically
        pass
        
    def _refresh_3d_plot(self):
        """Refresh the current 3D plot with new visualization settings"""
        if not hasattr(self, 'motion_frames') or not self.motion_frames:
            return
            
        try:
            frame_idx = self.current_frame_idx % len(self.motion_frames)
            positions = self.motion_frames[frame_idx]
            
            self.motion_ax.clear()
            self._plot_atoms_3d(positions, getattr(self, 'motion_atom_types', None))
            
            self.motion_ax.set_xlabel('X (Å)', fontsize=12)
            self.motion_ax.set_ylabel('Y (Å)', fontsize=12)
            self.motion_ax.set_zlabel('Z (Å)', fontsize=12)
            
            # Show current FPS in title for user feedback
            current_fps = self.anim_fps_var.get() if hasattr(self, 'anim_fps_var') else 10.0
            self.motion_ax.set_title(f'Atomic Motion Animation (Frame {frame_idx + 1}/{len(self.motion_frames)}) - {current_fps:.1f} FPS', fontsize=14)
            
            self._set_3d_view_limits(positions)
            self.motion_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error refreshing 3D plot: {e}")
            
    def _toggle_animation(self):
        """Toggle animation play/pause"""
        if self.animation_running:
            # Stop animation
            self.animation_running = False
            if hasattr(self, 'current_animation_id') and self.current_animation_id:
                self.root.after_cancel(self.current_animation_id)
                self.current_animation_id = None
            self.play_pause_button.config(text="▶️ Play")
        else:
            # Start animation
            self.animation_running = True
            self.play_pause_button.config(text="⏸️ Pause")
            if hasattr(self, 'motion_frames') and self.motion_frames:
                self._start_motion_animation()
                
    def _reset_3d_view(self):
        """Reset the 3D view to default angle"""
        if hasattr(self, 'motion_ax'):
            self.motion_ax.view_init(elev=20, azim=45)
            # Also reset any zoom/pan by recalculating limits
            if hasattr(self, 'motion_frames') and self.motion_frames:
                current_frame = self.current_frame_idx % len(self.motion_frames) if hasattr(self, 'current_frame_idx') else 0
                if current_frame < len(self.motion_frames):
                    self._set_3d_view_limits(self.motion_frames[current_frame])
            self.motion_canvas.draw()
        
    def _view_atomic_motion(self):
        """Open external viewer for atomic motion"""
        if not self.ised_result_path:
            messagebox.showwarning("Warning", "No iSED reconstruction available")
            return
            
        try:
            file_path = Path(self.ised_result_path)
            
            # Check if it's a temporary file
            is_temp = "/tmp" in str(file_path) or "temp" in str(file_path).lower()
            
            # Try to open with OVITO if available
            if self._check_ovito():
                subprocess.Popen(['ovito', self.ised_result_path])
                if is_temp:
                    messagebox.showinfo("External Viewer", 
                                      f"Opening temporary file {file_path.name} in OVITO...\n\n"
                                      "Note: This is a temporary file. Use the permanent save option "
                                      "if you want to keep this reconstruction.")
                else:
                    messagebox.showinfo("External Viewer", f"Opening {file_path.name} in OVITO...")
            else:
                # Fallback: show file location and open containing folder
                if is_temp:
                    messagebox.showinfo("Atomic Motion File", 
                                      f"Temporary iSED reconstruction:\n{file_path}\n\n"
                                      "This file will be deleted when the session ends.\n"
                                      "Use the permanent save option to keep reconstructions.\n\n"
                                      "You can open this file with OVITO, VMD, or other visualization software.")
                else:
                    messagebox.showinfo("Atomic Motion File", 
                                      f"iSED reconstruction saved to:\n{file_path}\n\n"
                                      f"Directory: {file_path.parent}\n\n"
                                      "You can open this file with OVITO, VMD, or other visualization software.")
                
                # Try to open the containing directory (only for permanent files)
                if not is_temp:
                    try:
                        if sys.platform == "darwin":  # macOS
                            subprocess.run(["open", str(file_path.parent)])
                        elif sys.platform == "win32":  # Windows
                            subprocess.run(["explorer", str(file_path.parent)])
                        else:  # Linux
                            subprocess.run(["xdg-open", str(file_path.parent)])
                    except Exception as e:
                        logger.debug(f"Could not open directory: {e}")
                
        except Exception as e:
            logger.error(f"Error opening atomic motion viewer: {e}")
            messagebox.showerror("Error", f"Failed to open viewer:\n{str(e)}")
            
    def _check_ovito(self) -> bool:
        """Check if OVITO is available"""
        try:
            subprocess.run(['ovito', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _start_motion_animation(self):
        """Start the atomic motion animation"""
        if not hasattr(self, 'motion_frames') or not self.motion_frames:
            return
            
        self.animation_running = True
        self.current_animation_id = None  # Track animation callback ID
        
        def animate():
            try:
                if not self.animation_running or not hasattr(self, 'motion_frames') or not self.motion_frames:
                    return
                    
                frame_idx = self.current_frame_idx % len(self.motion_frames)
                positions = self.motion_frames[frame_idx]
                
                self.motion_ax.clear()
                
                # Use new plotting method with visualization controls
                self._plot_atoms_3d(positions, getattr(self, 'motion_atom_types', None))
                
                self.motion_ax.set_xlabel('X (Å)', fontsize=12)
                self.motion_ax.set_ylabel('Y (Å)', fontsize=12)
                self.motion_ax.set_zlabel('Z (Å)', fontsize=12)
                
                # Show current FPS in title for user feedback
                current_fps = self.anim_fps_var.get() if hasattr(self, 'anim_fps_var') else 10.0
                self.motion_ax.set_title(f'Atomic Motion Animation (Frame {frame_idx + 1}/{len(self.motion_frames)}) - {current_fps:.1f} FPS', fontsize=14)
                
                # Maintain consistent view limits across all frames
                if len(self.motion_frames) > 1:
                    # Use limits based on all frames to prevent jitter
                    all_positions = np.vstack(self.motion_frames)
                    self._set_3d_view_limits(all_positions)
                else:
                    self._set_3d_view_limits(positions)
                
                self.motion_canvas.draw()
                
                self.current_frame_idx += 1
                
                # Schedule next frame if animation is still running
                if self.animation_running:
                    # Get current animation speed setting
                    try:
                        speed_fps = max(0.1, self.anim_fps_var.get())  # Prevent division by zero
                        delay_ms = int(1000 / speed_fps)
                    except (ValueError, ZeroDivisionError):
                        delay_ms = 100  # Default 10 FPS fallback
                    
                    self.current_animation_id = self.root.after(delay_ms, animate)
                    
            except Exception as e:
                logger.error(f"Error in animation: {e}")
                self.animation_running = False
                
        animate()

    def _toggle_save_mode(self):
        """Toggle between temp and permanent save mode"""
        if self.save_permanently_var.get():
            # Show output directory controls
            self.output_frame.pack(fill="x", pady=(0,10))
        else:
            # Hide output directory controls
            self.output_frame.pack_forget()

    def _cleanup_temp_files(self):
        """Clean up temporary iSED directory if it exists."""
        logger.info("Cleaning up temporary files...")
        if hasattr(self, '_current_temp_dir_obj') and self._current_temp_dir_obj:
            try:
                self._current_temp_dir_obj.cleanup()
                logger.info(f"Successfully cleaned up temp directory: {self._current_temp_dir_obj.name}")
                self.current_temp_ised_dir = None
                self._current_temp_dir_obj = None
            except Exception as e:
                logger.error(f"Error cleaning up temp directory {getattr(self._current_temp_dir_obj, 'name', 'N/A')}: {e}")
        elif self.current_temp_ised_dir: # Fallback if object wasn't stored but path was
             try:
                if Path(self.current_temp_ised_dir).exists():
                    import shutil
                    shutil.rmtree(self.current_temp_ised_dir)
                    logger.info(f"Successfully cleaned up temp directory (fallback): {self.current_temp_ised_dir}")
                self.current_temp_ised_dir = None
             except Exception as e:
                logger.error(f"Error cleaning up temp directory {self.current_temp_ised_dir} (fallback): {e}")

    def _toggle_chiral_options(self):
        """Toggle chirality sub-options display"""
        if self.chiral_sed_var.get():
            self.chiral_options_frame.pack(fill="x", pady=(0,10))
        else:
            self.chiral_options_frame.pack_forget()

    def _update_viz_controls_state(self):
        """Update state of visualization controls based on calculation mode and data."""
        if not hasattr(self, 'sed_calculation_type'):
            return
            
        calc_type = self.sed_calculation_type
        
        # Handle chiral toggle button for both K-Path and K-Grid
        has_phase_data = self.sed_result is not None and hasattr(self.sed_result, 'phase') and self.sed_result.phase is not None
        if has_phase_data:
            # Show the chiral toggle button before the save section
            self.plot_chiral_frame.pack(fill="x", pady=(0,5), before=self.save_section)
            
            # If toggle is currently enabled, show phase colormap
            if self.plot_chiral_var.get():
                self.phase_colormap_frame.pack(fill="x", pady=(0,5), before=self.save_section)
            else:
                self.phase_colormap_frame.pack_forget()
        else:
            # Hide chiral controls if no chiral data
            self.plot_chiral_frame.pack_forget()
            self.phase_colormap_frame.pack_forget()
            self.plot_chiral_var.set(False)  # Reset toggle state
        
        if calc_type == "K-Path":
            # Hide K-Grid specific controls for K-Path
            self.global_scale_frame.pack_forget()
                
        else:  # K-Grid
            # Show K-Grid specific controls only when not in chiral mode
            # Global scaling doesn't apply to phase data
            if has_phase_data and self.plot_chiral_var.get():
                self.global_scale_frame.pack_forget()
            else:
                self.global_scale_frame.pack(fill="x", pady=(0,5), before=self.save_section)

        # Plot button state depends if any SED result is available
        if hasattr(self, 'sed_result') and self.sed_result:
            self.plot_button.config(state="normal")
            # Enable save data button if we have plot data
            self.save_data_button.config(state="normal")
            # Enable save plot image button if we have plot data
            self.save_plot_image_button.config(state="normal")
            
            # Enable k-grid gif button only for k-grid mode with data
            if calc_type == "K-Grid" and hasattr(self, 'kgrid_sed_data') and self.kgrid_sed_data is not None:
                self.save_kgrid_gif_button.config(state="normal")
            else:
                self.save_kgrid_gif_button.config(state="disabled")
        else:
            self.plot_button.config(state="disabled")
            self.save_data_button.config(state="disabled")
            self.save_plot_image_button.config(state="disabled")
            self.save_kgrid_gif_button.config(state="disabled")
            
        # Enable save iSED button if we have iSED data
        if hasattr(self, 'ised_result_path') and self.ised_result_path:
            self.save_ised_button.config(state="normal")
        else:
            self.save_ised_button.config(state="disabled")

    def _validate_int_input(self, P):
        """Validate that the input P is an integer or empty. Allows positive integers."""
        if P == "":
            return True 
        try:
            val = int(P)
            if val >= 0: # Or val > 0 if strictly positive k-points needed
                return True
            else:
                self.root.bell()
                return False
        except ValueError:
            self.root.bell()
            return False

    def _on_intensity_colormap_change(self):
        """Update plot if Intensity colormap is changed."""
        if not hasattr(self, 'sed_calculation_type'):
            return
            
        if self.sed_calculation_type == "K-Path":
            # For k-path, regenerate the plot
            self._generate_sed_plot()
        elif self.sed_calculation_type == "K-Grid":
            # For k-grid, regenerate the heatmap at current frequency
            if hasattr(self, 'kgrid_freq_slider') and hasattr(self, 'kgrid_freqs'):
                current_freq = self.kgrid_freq_slider.get()
                freq_idx = np.argmin(np.abs(self.kgrid_freqs - current_freq))
                self._plot_kgrid_heatmap(freq_idx)

    def _on_intensity_scale_change(self):
        """Update plot when intensity scaling is changed."""
        # Clear cached global scaling values when intensity scaling changes
        if hasattr(self, '_cached_global_vmin'):
            delattr(self, '_cached_global_vmin')
        if hasattr(self, '_cached_global_vmax'):
            delattr(self, '_cached_global_vmax')
        if hasattr(self, '_cached_scale_type'):
            delattr(self, '_cached_scale_type')
            
        if not hasattr(self, 'sed_calculation_type'):
            return
            
        if self.sed_calculation_type == "K-Path":
            # For k-path, regenerate based on current mode
            self._generate_sed_plot()
        elif self.sed_calculation_type == "K-Grid":
            # For k-grid, regenerate the heatmap at current frequency
            if hasattr(self, 'kgrid_freq_slider') and hasattr(self, 'kgrid_freqs'):
                current_freq = self.kgrid_freq_slider.get()
                freq_idx = np.argmin(np.abs(self.kgrid_freqs - current_freq))
                self._plot_kgrid_heatmap(freq_idx)

    def _on_plot_chiral_toggle(self):
        """Handle plot chiral toggle button changes."""
        if self.plot_chiral_var.get():
            # Show phase colormap when chiral is enabled
            self.phase_colormap_frame.pack(fill="x", pady=(0,5), before=self.save_section)
        else:
            # Hide phase colormap when chiral is disabled
            self.phase_colormap_frame.pack_forget()
        
        # Update global scaling visibility for K-Grid mode
        if hasattr(self, 'sed_calculation_type') and self.sed_calculation_type == "K-Grid":
            has_phase_data = self.sed_result is not None and hasattr(self.sed_result, 'phase') and self.sed_result.phase is not None
            if has_phase_data and self.plot_chiral_var.get():
                # Hide global scaling when in chiral mode
                self.global_scale_frame.pack_forget()
            else:
                # Show global scaling when in SED mode
                self.global_scale_frame.pack(fill="x", pady=(0,5), before=self.save_section)
        
        # Immediately update the plot if we have data
        if hasattr(self, 'sed_calculation_type'):
            if self.sed_calculation_type == "K-Path":
                self._generate_sed_plot()
            elif self.sed_calculation_type == "K-Grid":
                # For k-grid, regenerate the heatmap at current frequency
                if hasattr(self, 'kgrid_freq_slider') and hasattr(self, 'kgrid_freqs'):
                    current_freq = self.kgrid_freq_slider.get()
                    freq_idx = np.argmin(np.abs(self.kgrid_freqs - current_freq))
                    self._plot_kgrid_heatmap(freq_idx)

    def _update_kgrid_axis_controls(self):
        """Update axis labels and entry states based on selected plane."""
        plane = self.kgrid_plane_var.get()
        # Reset all to normal
        for key in self.kgrid_axis_entries:
            self.kgrid_axis_entries[key].config(state="normal")
            self.kgrid_axis_labels[key].config(foreground="black")
        # Hide all
        for key in self.kgrid_axis_labels:
            self.kgrid_axis_labels[key].grid_remove()
        for key in self.kgrid_axis_entries:
            self.kgrid_axis_entries[key].grid_remove()
        
        # Update n_kx and n_ky labels based on plane
        if plane == "xy":
            self.kgrid_n_kx_label.config(text="n_kx:")
            self.kgrid_n_ky_label.config(text="n_ky:")
        elif plane == "yz":
            self.kgrid_n_kx_label.config(text="n_ky:")
            self.kgrid_n_ky_label.config(text="n_kz:")
        elif plane == "zx":
            self.kgrid_n_kx_label.config(text="n_kz:")
            self.kgrid_n_ky_label.config(text="n_kx:")
        
        # Position widgets based on plane
        if plane == "xy":
            # Show kx min/max, ky min/max, kz (fixed)
            self.kgrid_axis_labels['kx_min'].config(text="kx min:")
            self.kgrid_axis_labels['kx_min'].grid(row=0, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['kx_min'].grid(row=0, column=1, sticky="w", padx=2)
            self.kgrid_axis_labels['kx_max'].config(text="kx max:")
            self.kgrid_axis_labels['kx_max'].grid(row=0, column=2, sticky="w", padx=2)
            self.kgrid_axis_entries['kx_max'].grid(row=0, column=3, sticky="w", padx=2)
            self.kgrid_axis_labels['ky_min'].config(text="ky min:")
            self.kgrid_axis_labels['ky_min'].grid(row=1, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['ky_min'].grid(row=1, column=1, sticky="w", padx=2)
            self.kgrid_axis_labels['ky_max'].config(text="ky max:")
            self.kgrid_axis_labels['ky_max'].grid(row=1, column=2, sticky="w", padx=2)
            self.kgrid_axis_entries['ky_max'].grid(row=1, column=3, sticky="w", padx=2)
            self.kgrid_axis_labels['kz_fixed'].config(text="kz (fixed):")
            self.kgrid_axis_labels['kz_fixed'].grid(row=2, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['kz_fixed'].grid(row=2, column=1, sticky="w", padx=2)
        elif plane == "yz":
            # Show ky min/max, kz min/max, kx (fixed)
            self.kgrid_axis_labels['ky_min'].config(text="ky min:")
            self.kgrid_axis_labels['ky_min'].grid(row=0, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['ky_min'].grid(row=0, column=1, sticky="w", padx=2)
            self.kgrid_axis_labels['ky_max'].config(text="ky max:")
            self.kgrid_axis_labels['ky_max'].grid(row=0, column=2, sticky="w", padx=2)
            self.kgrid_axis_entries['ky_max'].grid(row=0, column=3, sticky="w", padx=2)
            self.kgrid_axis_labels['kz_fixed'].config(text="kz min:")
            self.kgrid_axis_labels['kz_fixed'].grid(row=1, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['kz_fixed'].grid(row=1, column=1, sticky="w", padx=2)
            self.kgrid_axis_labels['kx_min'].config(text="kz max:")
            self.kgrid_axis_labels['kx_min'].grid(row=1, column=2, sticky="w", padx=2)
            self.kgrid_axis_entries['kx_min'].grid(row=1, column=3, sticky="w", padx=2)
            self.kgrid_axis_labels['kx_max'].config(text="kx (fixed):")
            self.kgrid_axis_labels['kx_max'].grid(row=2, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['kx_max'].grid(row=2, column=1, sticky="w", padx=2)
        elif plane == "zx":
            # Show kz min/max, kx min/max, ky (fixed)
            self.kgrid_axis_labels['kz_fixed'].config(text="kz min:")
            self.kgrid_axis_labels['kz_fixed'].grid(row=0, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['kz_fixed'].grid(row=0, column=1, sticky="w", padx=2)
            self.kgrid_axis_labels['kx_min'].config(text="kz max:")
            self.kgrid_axis_labels['kx_min'].grid(row=0, column=2, sticky="w", padx=2)
            self.kgrid_axis_entries['kx_min'].grid(row=0, column=3, sticky="w", padx=2)
            self.kgrid_axis_labels['ky_min'].config(text="kx min:")
            self.kgrid_axis_labels['ky_min'].grid(row=1, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['ky_min'].grid(row=1, column=1, sticky="w", padx=2)
            self.kgrid_axis_labels['ky_max'].config(text="kx max:")
            self.kgrid_axis_labels['ky_max'].grid(row=1, column=2, sticky="w", padx=2)
            self.kgrid_axis_entries['ky_max'].grid(row=1, column=3, sticky="w", padx=2)
            self.kgrid_axis_labels['kx_max'].config(text="ky (fixed):")
            self.kgrid_axis_labels['kx_max'].grid(row=2, column=0, sticky="w", padx=2)
            self.kgrid_axis_entries['kx_max'].grid(row=2, column=1, sticky="w", padx=2)

    def _toggle_calc_mode(self):
        """Toggle between K-Path and K-Grid parameter display"""
        mode = self.calc_mode_var.get()
        if mode == "K-Path":
            self.kpath_frame.pack(fill="x", pady=(0,10), before=self.sed_spacer_frame)
            self.kgrid_frame.pack_forget()
        else:  # K-Grid
            self.kpath_frame.pack_forget()
            self.kgrid_frame.pack(fill="x", pady=(0,10), before=self.sed_spacer_frame)
            self._update_kgrid_axis_controls()
        # Update visualization controls if SED is already calculated
        if hasattr(self, 'sed_result') and self.sed_result:
            self._update_viz_controls_state()

    def _generate_plot(self):
        """Generate plot based on calculation type"""
        if not hasattr(self, 'sed_calculation_type'):
            messagebox.showerror("Error", "Please calculate SED first")
            return
            
        if self.sed_calculation_type == "K-Path":
            # Setup dispersion plot
            self._setup_dispersion_plot()
            self._generate_sed_plot()
        else:  # K-Grid
            # Setup heatmap plot (this handles slider setup with proper frequency values)
            self._setup_heatmap_plot()
            if hasattr(self, 'kgrid_freqs') and len(self.kgrid_freqs) > 0:
                # Plot the first frequency
                self._plot_kgrid_heatmap(0)
            else:
                messagebox.showerror("Error", "No positive frequencies found in k-grid data")

    def _calculate_kgrid_sed(self):
        """Calculate K-Grid SED"""
        def calc_worker():
            try:
                self.root.after(0, lambda: self.sed_status_var.set("Calculating K-Grid SED..."))
                
                plane = self.kgrid_plane_var.get()
                n_kx = self.kgrid_n_kx_var.get()
                n_ky = self.kgrid_n_ky_var.get()
                kx_min = self.kgrid_kx_min_var.get()
                kx_max = self.kgrid_kx_max_var.get()
                ky_min = self.kgrid_ky_min_var.get()
                ky_max = self.kgrid_ky_max_var.get()
                kz_val = self.kgrid_kz_val_var.get()
                
                # Get k-grid
                if plane == "xy":
                    k_range_x = (kx_min, kx_max)
                    k_range_y = (ky_min, ky_max)
                    k_fixed_val = kz_val
                elif plane == "yz":
                    # For yz plane: x-axis is ky, y-axis is kz, fixed is kx
                    # Widget mapping: ky_min/max->ky range, kz_fixed/kx_min->kz range, kx_max->kx fixed
                    k_range_x = (ky_min, ky_max)  # ky range maps to x-axis of the plane
                    k_range_y = (kz_val, kx_min)  # kz range (kz_fixed to kx_min) maps to y-axis 
                    k_fixed_val = kx_max  # kx value (stored in kx_max entry)
                elif plane == "zx":
                    # For zx plane: x-axis is kz, y-axis is kx, fixed is ky  
                    # Widget mapping: kz_fixed/kx_min->kz range, ky_min/max->kx range, kx_max->ky fixed
                    k_range_x = (kz_val, kx_min)  # kz range (kz_fixed to kx_min) maps to x-axis
                    k_range_y = (ky_min, ky_max)  # kx range (stored in ky_min/max) maps to y-axis
                    k_fixed_val = kx_max  # ky value (stored in kx_max entry)
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Invalid plane: {plane}"))
                    return
                    
                k_mags, k_vecs, k_grid_shape = self.sed_calculator.get_k_grid(
                    plane=plane,
                    k_range_x=k_range_x,
                    k_range_y=k_range_y,
                    n_kx=n_kx,
                    n_ky=n_ky,
                    k_fixed_val=k_fixed_val
                )
                
                basis_types = self._get_basis_atom_types()
                
                # Handle chirality calculation for K-Grid
                summation_mode_to_use = self.summation_mode_var.get()
                if self.chiral_sed_var.get():
                    if summation_mode_to_use != 'coherent':
                        logger.info("Chirality calculation selected for K-Grid, forcing coherent summation mode.")
                        summation_mode_to_use = 'coherent'
                
                sed_obj = self.sed_calculator.calculate(
                    k_points_mags=k_mags,
                    k_vectors_3d=k_vecs,
                    k_grid_shape=k_grid_shape,
                    basis_atom_types=basis_types,
                    summation_mode=summation_mode_to_use
                )
                
                # Calculate phase for K-Grid if requested
                calculated_phase = None
                if self.chiral_sed_var.get():
                    if hasattr(sed_obj, 'sed') and sed_obj.sed is not None and sed_obj.is_complex:
                        if sed_obj.sed.ndim == 3 and sed_obj.sed.shape[-1] >= 2:
                            axis = self.chiral_axis_var.get()
                            if axis == 'x':
                                idx1, idx2 = 1, 2
                            elif axis == 'y':
                                idx1, idx2 = 0, 2
                            else:
                                idx1, idx2 = 0, 1
                            # For K-Grid, we need to calculate phase for each frequency slice
                            Z1 = sed_obj.sed[:,:,idx1]
                            Z2 = sed_obj.sed[:,:,idx2]
                            logger.info(f"Calculating chiral phase for K-Grid using axis {axis} (components {idx1} and {idx2}).")
                            calculated_phase = self.sed_calculator.calculate_chiral_phase(
                                Z1=Z1, Z2=Z2, angle_range_opt="C"
                            )
                            logger.info("K-Grid chiral phase calculation complete.")
                
                # Store the SED result for the plotting
                self.sed_result = SED(
                    sed=sed_obj.sed,
                    freqs=sed_obj.freqs,
                    k_points=sed_obj.k_points,
                    k_vectors=sed_obj.k_vectors,
                    k_grid_shape=sed_obj.k_grid_shape,
                    phase=calculated_phase,  # Now includes phase for K-Grid if calculated
                    is_complex=sed_obj.is_complex
                )
                
                # Store k-grid specific data with max frequency filtering
                self.kgrid_sed_result = sed_obj
                all_freqs = sed_obj.freqs
                pos_mask = all_freqs >= 0
                kgrid_freqs_unfiltered = all_freqs[pos_mask]
                kgrid_sed_data_unfiltered = sed_obj.sed[pos_mask] if sed_obj.sed is not None else None
                
                # Apply max frequency filter if specified
                if self.max_freq_var.get().strip():
                    try:
                        max_freq_val = float(self.max_freq_var.get())
                        max_freq_mask = kgrid_freqs_unfiltered <= max_freq_val
                        self.kgrid_freqs = kgrid_freqs_unfiltered[max_freq_mask]
                        self.kgrid_sed_data = kgrid_sed_data_unfiltered[max_freq_mask] if kgrid_sed_data_unfiltered is not None else None
                    except ValueError:
                        # If max frequency is invalid, use all positive frequencies
                        self.kgrid_freqs = kgrid_freqs_unfiltered
                        self.kgrid_sed_data = kgrid_sed_data_unfiltered
                else:
                    # No max frequency specified, use all positive frequencies
                    self.kgrid_freqs = kgrid_freqs_unfiltered
                    self.kgrid_sed_data = kgrid_sed_data_unfiltered
                
                # Cache kx/ky axes for plotting
                k_vectors = sed_obj.k_vectors
                if plane == "xy":
                    self.kgrid_kx = np.unique(k_vectors[:,0])
                    self.kgrid_ky = np.unique(k_vectors[:,1])
                    self.kgrid_xlabel = r'$k_x$ ($2\pi/\AA$)'
                    self.kgrid_ylabel = r'$k_y$ ($2\pi/\AA$)'
                elif plane == "yz":
                    self.kgrid_kx = np.unique(k_vectors[:,1])
                    self.kgrid_ky = np.unique(k_vectors[:,2])
                    self.kgrid_xlabel = r'$k_y$ ($2\pi/\AA$)'
                    self.kgrid_ylabel = r'$k_z$ ($2\pi/\AA$)'
                elif plane == "zx":
                    self.kgrid_kx = np.unique(k_vectors[:,2])
                    self.kgrid_ky = np.unique(k_vectors[:,0])
                    self.kgrid_xlabel = r'$k_z$ ($2\pi/\AA$)'
                    self.kgrid_ylabel = r'$k_x$ ($2\pi/\AA$)'
                
                # Store calculation type
                self.sed_calculation_type = "K-Grid"
                
                self.root.after(0, lambda: self.sed_status_var.set("K-Grid SED calculation complete"))
                self.root.after(0, lambda: self.plot_button.config(state="normal"))
                self.root.after(0, self._update_viz_controls_state)
                
            except Exception as e:
                logger.error(f"Error calculating K-Grid SED: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate k-grid SED:\n{e}"))
                self.root.after(0, lambda: self.sed_status_var.set("Error calculating K-Grid SED"))
        
        thread = threading.Thread(target=calc_worker)
        thread.daemon = True
        thread.start()

    def _setup_dispersion_plot(self):
        """Setup the dispersion plot in the SED analysis tab"""
        # Switch to Reciprocal Space tab
        self.plot_notebook.select(0)
        
        # Hide frequency slider if it exists (used for k-grid plots)
        if hasattr(self, 'freq_slider_frame'):
            self.freq_slider_frame.pack_forget()

    def _setup_heatmap_plot(self):
        """Setup the k-grid heatmap plot in the SED analysis tab"""
        # Switch to Reciprocal Space tab
        self.plot_notebook.select(0)
        
        # Apply max frequency filtering to existing k-grid data
        if hasattr(self, 'kgrid_sed_result') and self.kgrid_sed_result:
            all_freqs = self.kgrid_sed_result.freqs
            pos_mask = all_freqs >= 0
            kgrid_freqs_unfiltered = all_freqs[pos_mask]
            kgrid_sed_data_unfiltered = self.kgrid_sed_result.sed[pos_mask] if self.kgrid_sed_result.sed is not None else None
            
            # Apply max frequency filter based on current Plot Parameters setting
            if self.max_freq_var.get().strip():
                try:
                    max_freq_val = float(self.max_freq_var.get())
                    max_freq_mask = kgrid_freqs_unfiltered <= max_freq_val
                    self.kgrid_freqs = kgrid_freqs_unfiltered[max_freq_mask]
                    self.kgrid_sed_data = kgrid_sed_data_unfiltered[max_freq_mask] if kgrid_sed_data_unfiltered is not None else None
                except ValueError:
                    # If max frequency is invalid, use all positive frequencies
                    self.kgrid_freqs = kgrid_freqs_unfiltered
                    self.kgrid_sed_data = kgrid_sed_data_unfiltered
            else:
                # No max frequency specified, use all positive frequencies
                self.kgrid_freqs = kgrid_freqs_unfiltered
                self.kgrid_sed_data = kgrid_sed_data_unfiltered
            
            # Clear cached global scaling when frequency range changes
            if hasattr(self, '_cached_global_vmin'):
                delattr(self, '_cached_global_vmin')
            if hasattr(self, '_cached_global_vmax'):
                delattr(self, '_cached_global_vmax')
            if hasattr(self, '_cached_scale_type'):
                delattr(self, '_cached_scale_type')
        
        # Enable frequency slider if k-grid data exists
        if hasattr(self, 'kgrid_freqs') and len(self.kgrid_freqs) > 0:
            # Create a frame for the frequency slider in the SED plot area if it doesn't exist
            if not hasattr(self, 'freq_slider_frame'):
                self.freq_slider_frame = ttk.Frame(self.sed_plot_frame)
                self.freq_slider_frame.pack(fill="x", side="top", before=self.sed_canvas.get_tk_widget())
                
                # Use actual frequency values instead of indices
                min_freq = float(self.kgrid_freqs.min())
                max_freq = float(self.kgrid_freqs.max())
                
                # Calculate proper frequency resolution for slider
                if len(self.kgrid_freqs) > 1:
                    # Use the median frequency difference for more robust estimation
                    freq_diffs = np.diff(self.kgrid_freqs)
                    freq_step = float(np.median(freq_diffs))
                    # Round to a reasonable number of significant figures
                    freq_step = round(freq_step, 6)
                else:
                    freq_step = 1.0
                
                self.kgrid_freq_slider = tk.Scale(self.freq_slider_frame, from_=min_freq, to=max_freq, 
                                                 orient="horizontal", resolution=freq_step, label="Frequency (THz)", 
                                                 command=self._on_kgrid_freq_slider, digits=4)
                self.kgrid_freq_slider.pack(fill="x", padx=10, pady=5)
                
                # No separate label needed since Scale shows the value
            else:
                # Update existing slider range based on (possibly filtered) frequencies
                min_freq = float(self.kgrid_freqs.min())
                max_freq = float(self.kgrid_freqs.max())
                
                # Calculate proper frequency resolution for slider
                if len(self.kgrid_freqs) > 1:
                    # Use the median frequency difference for more robust estimation
                    freq_diffs = np.diff(self.kgrid_freqs)
                    freq_step = float(np.median(freq_diffs))
                    # Round to a reasonable number of significant figures
                    freq_step = round(freq_step, 6)
                else:
                    freq_step = 1.0
                
                self.kgrid_freq_slider.config(from_=min_freq, to=max_freq, resolution=freq_step, state="normal")
            
            # Set initial frequency and show slider
            self.kgrid_freq_slider.set(float(self.kgrid_freqs[0]))
            self.freq_slider_frame.pack(fill="x", side="top", before=self.sed_canvas.get_tk_widget())
            
        else:
            if hasattr(self, 'freq_slider_frame'):
                self.freq_slider_frame.pack_forget()

    def _on_kgrid_freq_slider(self, val):
        """Handle frequency slider change for k-grid heatmap"""
        if hasattr(self, 'kgrid_freqs') and self.kgrid_freqs is not None:
            freq_val = float(val)
            # Find the closest frequency index
            freq_idx = np.argmin(np.abs(self.kgrid_freqs - freq_val))
            
            # Plot the heatmap for this frequency
            self._plot_kgrid_heatmap(freq_idx)

    def _plot_kgrid_heatmap(self, freq_idx):
        """Plot k-grid heatmap at given frequency index in the unified SED plot area"""
        if not hasattr(self, 'sed_ax') or not hasattr(self, 'kgrid_sed_result'):
            return
            
        self.sed_ax.clear()
        # Remove previous colorbar and its axes if they exist
        self._cleanup_colorbar()
            
        if not self.kgrid_sed_result or self.kgrid_freqs is None or self.kgrid_sed_data is None:
            self.sed_ax.set_title("No k-grid data")
            self.sed_canvas.draw()
            return
            
        try:
            # Get intensity or phase at this frequency based on chiral toggle
            sed = self.kgrid_sed_result
            freq_val = self.kgrid_freqs[int(freq_idx)]
            
            # Check if we should plot chiral/phase data
            plot_chiral = self.plot_chiral_var.get()
            has_phase_data = hasattr(self.sed_result, 'phase') and self.sed_result.phase is not None
            
            if plot_chiral and has_phase_data:
                # Plot phase data
                intensity = self.sed_result.phase[int(freq_idx), :]
                colorbar_label = "Phase Angle (radians)"
                plot_title = f"Chirality @ {freq_val:.2f} THz"
                effective_cmap = self.phase_colormap_var.get()
                # Don't apply intensity scaling to phase data
            else:
                # Plot SED intensity
                if sed.is_complex:
                    intensity = np.sum(np.abs(self.kgrid_sed_data[int(freq_idx), :, :])**2, axis=-1)
                else:
                    if self.kgrid_sed_data.ndim == 3:
                        intensity = np.sum(self.kgrid_sed_data[int(freq_idx), :, :], axis=-1)
                    elif self.kgrid_sed_data.ndim == 2:
                        intensity = self.kgrid_sed_data[int(freq_idx), :]
                    else:
                        self.sed_ax.set_title("Unsupported SED shape")
                        self.sed_canvas.draw()
                        return
                
                # Apply intensity scaling only for SED data
                intensity = self._apply_intensity_scaling(intensity, self.intensity_scale_var.get())
                colorbar_label = "Intensity"
                plot_title = f"SED @ {freq_val:.2f} THz"
                effective_cmap = self.colormap_var.get()
                    
            n_kx = len(self.kgrid_kx)
            n_ky = len(self.kgrid_ky)
            intensity_grid = intensity.reshape(n_kx, n_ky).T # Transpose for correct orientation
            X, Y = np.meshgrid(self.kgrid_kx, self.kgrid_ky)
            
            # Global scaling with caching for performance (only for SED, not phase)
            vmin = vmax = None
            if not plot_chiral and getattr(self, 'kgrid_global_scale_var', None) and self.kgrid_global_scale_var.get():
                # Check if we have cached values for the current scale type
                current_scale_type = self.intensity_scale_var.get()
                if (hasattr(self, '_cached_global_vmin') and hasattr(self, '_cached_global_vmax') and 
                    hasattr(self, '_cached_scale_type') and self._cached_scale_type == current_scale_type):
                    # Use cached values
                    vmin = self._cached_global_vmin
                    vmax = self._cached_global_vmax
                else:
                    # Compute and cache global min/max across all freq slices
                    if sed.is_complex:
                        all_intensity = np.sum(np.abs(self.kgrid_sed_data)**2, axis=-1)
                    else:
                        if self.kgrid_sed_data.ndim == 3:
                            all_intensity = np.sum(self.kgrid_sed_data, axis=-1)
                        elif self.kgrid_sed_data.ndim == 2:
                            all_intensity = self.kgrid_sed_data
                        else:
                            all_intensity = None
                    
                    if all_intensity is not None:
                        # Apply scaling to all_intensity
                        all_intensity = self._apply_intensity_scaling(all_intensity, current_scale_type)
                        
                        # Cache the results
                        self._cached_global_vmin = vmin = np.nanmin(all_intensity)
                        self._cached_global_vmax = vmax = np.nanmax(all_intensity)
                        self._cached_scale_type = current_scale_type
            
            pcm = self.sed_ax.pcolormesh(X, Y, intensity_grid, cmap=effective_cmap, shading='auto', vmin=vmin, vmax=vmax)
            self.sed_ax.set_xlabel(self.kgrid_xlabel)
            self.sed_ax.set_ylabel(self.kgrid_ylabel)
            self.sed_ax.set_title(plot_title)
            
            # Add colorbar using make_axes_locatable
            divider = make_axes_locatable(self.sed_ax)
            self.sed_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
            self.sed_colorbar_ax.clear()
            self.sed_colorbar = self.sed_fig.colorbar(pcm, cax=self.sed_colorbar_ax)
            self.sed_colorbar.set_label(colorbar_label, fontsize=12)
            
            self.sed_canvas.draw()
            
        except Exception as e:
            self.sed_ax.set_title(f"Plot error: {e}")
            self.sed_canvas.draw()

    def _on_max_freq_change(self):
        """Handle change in max frequency entry"""
        # Only update if we're in k-grid mode and have calculated k-grid data
        if (hasattr(self, 'sed_calculation_type') and self.sed_calculation_type == "K-Grid" and 
            hasattr(self, 'kgrid_sed_result') and self.kgrid_sed_result):
            # Re-setup the heatmap plot with new frequency filtering
            self._setup_heatmap_plot()
            # Plot the first frequency if we have data
            if hasattr(self, 'kgrid_freqs') and len(self.kgrid_freqs) > 0:
                self._plot_kgrid_heatmap(0)

    def _save_plot_data(self):
        """Save current plot data as numpy or CSV files"""
        try:
            if not hasattr(self, 'sed_result') or not self.sed_result:
                messagebox.showerror("Error", "No plot data available to save")
                return
                
            # Create output directory
            output_dir = Path(self.output_dir_var.get())
            output_dir.mkdir(parents=True, exist_ok=True)
            
            format_type = self.save_format_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if hasattr(self, 'sed_calculation_type'):
                calc_type = self.sed_calculation_type.lower().replace("-", "_")
            else:
                calc_type = "unknown"
            
            # Use custom filename if provided, otherwise generate one
            custom_name = self.custom_data_filename_var.get().strip()
            if custom_name:
                # Remove extension if user provided one
                custom_name = Path(custom_name).stem
                base_filename = custom_name
            else:
                base_filename = f"psa_{calc_type}_data_{timestamp}"
            
            if format_type == "npy":
                # Save as numpy arrays
                
                # Save frequencies
                freq_file = output_dir / f"{base_filename}_frequencies.npy"
                np.save(freq_file, self.sed_result.freqs)
                
                # Save SED data
                sed_file = output_dir / f"{base_filename}_sed.npy"
                np.save(sed_file, self.sed_result.sed)
                
                # Save k-points/k-vectors
                k_points_file = output_dir / f"{base_filename}_k_points.npy"
                if hasattr(self.sed_result, 'k_points') and self.sed_result.k_points is not None:
                    np.save(k_points_file, self.sed_result.k_points)
                elif hasattr(self.sed_result, 'k_vectors') and self.sed_result.k_vectors is not None:
                    np.save(k_points_file, self.sed_result.k_vectors)
                
                # Save phase data if available (chirality)
                if hasattr(self.sed_result, 'phase') and self.sed_result.phase is not None:
                    phase_file = output_dir / f"{base_filename}_phase.npy"
                    np.save(phase_file, self.sed_result.phase)
                
                saved_files = [freq_file, sed_file, k_points_file]
                if hasattr(self.sed_result, 'phase') and self.sed_result.phase is not None:
                    saved_files.append(phase_file)
                    
                messagebox.showinfo("Save Complete", 
                                   f"Plot data saved as .npy files:\n" + 
                                   "\n".join([f.name for f in saved_files]))
                                   
            elif format_type == "csv":
                # Save as CSV files
                
                # For k-path data, create a single CSV with columns
                if calc_type == "k_path":
                    csv_file = output_dir / f"{base_filename}.csv"
                    
                    # Prepare data - frequencies are rows, k-points are columns
                    try:
                        import pandas as pd
                    except ImportError:
                        messagebox.showerror("Error", "pandas is required for CSV export. Please install it with: pip install pandas")
                        return

                    freqs = self.sed_result.freqs
                    sed_data = self.sed_result.sed
                    k_points = self.sed_result.k_points if hasattr(self.sed_result, 'k_points') else np.arange(sed_data.shape[1])
                    
                    # Create comprehensive CSV with metadata and all data
                    with open(csv_file, 'w') as f:
                        # Write metadata header
                        f.write("# PSA K-Path SED Data Export\n")
                        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"# Calculation Type: {calc_type.upper()}\n")
                        f.write(f"# Data Shape: {sed_data.shape}\n")
                        f.write(f"# Is Complex: {getattr(self.sed_result, 'is_complex', False)}\n")
                        f.write(f"# Direction: {self.direction_var.get()}\n")
                        f.write(f"# BZ Coverage: {self.bz_coverage_var.get()}\n")
                        f.write(f"# Frequency Range: {freqs.min():.3f} to {freqs.max():.3f} THz\n")
                        f.write("#\n")
                    
                    # Handle 3D SED data (freq, k-points, components)
                    if sed_data.ndim == 3:
                        # Create a multi-level CSV with all components
                        data_dict = {'Frequency_THz': freqs}
                        
                        # Add k-point coordinates
                        for i, k_val in enumerate(k_points):
                            # Total intensity (summed over components)
                            if hasattr(self.sed_result, 'is_complex') and self.sed_result.is_complex:
                                total_intensity = np.sum(np.abs(sed_data[:, i, :])**2, axis=-1)
                            else:
                                total_intensity = np.sum(sed_data[:, i, :], axis=-1)
                            data_dict[f"k{i:03d}_total"] = total_intensity
                            
                            # Individual components
                            for comp_idx in range(sed_data.shape[-1]):
                                comp_data = sed_data[:, i, comp_idx]
                                if hasattr(self.sed_result, 'is_complex') and self.sed_result.is_complex:
                                    comp_data = np.abs(comp_data)**2
                                data_dict[f"k{i:03d}_comp{comp_idx}"] = comp_data
                    
                    elif sed_data.ndim == 2:
                        # 2D data - frequencies vs k-points
                        data_dict = {'Frequency_THz': freqs}
                        for i, k_val in enumerate(k_points):
                            data_dict[f"k{i:03d}_{k_val:.3f}"] = sed_data[:, i]
                    
                    else:
                        # 1D or other, convert to 2D
                        reshaped_data = sed_data.reshape(len(freqs), -1)
                        data_dict = {'Frequency_THz': freqs}
                        for i in range(reshaped_data.shape[1]):
                            data_dict[f"k{i:03d}"] = reshaped_data[:, i]
                    
                    # Create DataFrame and append to CSV
                    df = pd.DataFrame(data_dict)
                    df.to_csv(csv_file, mode='a', index=False)
                    
                    messagebox.showinfo("Save Complete", f"K-Path data saved as comprehensive CSV:\n{csv_file.name}")
                    
                else:  # k-grid data
                    csv_file = output_dir / f"{base_filename}.csv"
                    
                    try:
                        import pandas as pd
                    except ImportError:
                        messagebox.showerror("Error", "pandas is required for CSV export. Please install it with: pip install pandas")
                        return

                    # Create comprehensive CSV for k-grid data
                    with open(csv_file, 'w') as f:
                        # Write metadata header
                        f.write("# PSA K-Grid SED Data Export\n")
                        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"# Calculation Type: {calc_type.upper()}\n")
                        f.write(f"# Data Shape: {self.sed_result.sed.shape}\n")
                        f.write(f"# K-Grid Shape: {getattr(self.sed_result, 'k_grid_shape', 'N/A')}\n")
                        f.write(f"# Frequency Range: {self.sed_result.freqs.min():.3f} to {self.sed_result.freqs.max():.3f} THz\n")
                        f.write(f"# Plane: {self.kgrid_plane_var.get()}\n")
                        f.write("#\n")
                    
                    # For k-grid, flatten the data into a single table
                    # Each row will be: frequency, kx, ky, kz, intensity
                    data_list = []
                    
                    freqs = self.sed_result.freqs
                    k_vectors = self.sed_result.k_vectors
                    
                    for freq_idx, freq_val in enumerate(freqs):
                        for k_idx, k_vec in enumerate(k_vectors):
                            # Get intensity at this frequency and k-point
                            if hasattr(self.sed_result, 'is_complex') and self.sed_result.is_complex:
                                if self.sed_result.sed.ndim == 3:
                                    intensity = np.sum(np.abs(self.sed_result.sed[freq_idx, k_idx, :])**2)
                                else:
                                    intensity = np.abs(self.sed_result.sed[freq_idx, k_idx])**2
                            else:
                                if self.sed_result.sed.ndim == 3:
                                    intensity = np.sum(self.sed_result.sed[freq_idx, k_idx, :])
                                else:
                                    intensity = self.sed_result.sed[freq_idx, k_idx]
                            
                            data_list.append({
                                'Frequency_THz': freq_val,
                                'kx': k_vec[0],
                                'ky': k_vec[1], 
                                'kz': k_vec[2],
                                'Intensity': intensity
                            })
                    
                    # Create DataFrame and append to CSV
                    df = pd.DataFrame(data_list)
                    df.to_csv(csv_file, mode='a', index=False)
                    
                    messagebox.showinfo("Save Complete", f"K-Grid data saved as comprehensive CSV:\n{csv_file.name}")
                                       
        except Exception as e:
            logger.error(f"Error saving plot data: {e}")
            messagebox.showerror("Error", f"Failed to save plot data:\n{str(e)}")

    def _save_kgrid_gif(self):
        """Save k-grid animation as GIF"""
        try:
            if not hasattr(self, 'kgrid_sed_data') or self.kgrid_sed_data is None:
                messagebox.showerror("Error", "No k-grid data available to save as animation")
                return
                
            if not hasattr(self, 'kgrid_freqs') or len(self.kgrid_freqs) == 0:
                messagebox.showerror("Error", "No frequency data available for animation")
                return
            
            # Create output directory
            output_dir = Path(self.output_dir_var.get())
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use custom filename if provided, otherwise generate one
            custom_name = self.custom_animation_filename_var.get().strip()
            if custom_name:
                # Remove extension if user provided one
                custom_name = Path(custom_name).stem
                gif_file = output_dir / f"{custom_name}.gif"
            else:
                gif_file = output_dir / f"kgrid_animation_{timestamp}.gif"
            
            # Show progress dialog
            progress = ProgressDialog(self.root, "Creating GIF Animation...")
            progress.update_message("Preparing frames...", f"Total frames: {len(self.kgrid_freqs)}")
            
            # Get user-specified DPI for consistent quality with image saving
            gif_dpi = self.plot_dpi_var.get()
            
            # Create temporary figure for animation frames with fixed layout
            temp_fig = Figure(figsize=(10, 8), dpi=gif_dpi)  # Use user-specified DPI
            temp_fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)  # Fixed layout
            temp_ax = temp_fig.add_subplot(111)
            
            frames = []
            total_frames = len(self.kgrid_freqs)
            
            # Pre-calculate global scaling if enabled
            vmin = vmax = None
            if getattr(self, 'kgrid_global_scale_var', None) and self.kgrid_global_scale_var.get():
                if hasattr(self, 'sed_result') and self.sed_result.is_complex:
                    all_intensity = np.sum(np.abs(self.kgrid_sed_data)**2, axis=-1)
                else:
                    if self.kgrid_sed_data.ndim == 3:
                        all_intensity = np.sum(self.kgrid_sed_data, axis=-1)
                    else:
                        all_intensity = self.kgrid_sed_data
                
                # Apply scaling
                scale_type = self.intensity_scale_var.get()
                if scale_type == "log":
                    all_intensity = np.log10(np.maximum(all_intensity, 1e-12))
                elif scale_type == "sqrt":
                    all_intensity = np.sqrt(np.maximum(all_intensity, 0))
                elif scale_type == "dsqrt":
                    all_intensity = np.sqrt(np.sqrt(np.maximum(all_intensity, 0)))
                
                vmin = np.nanmin(all_intensity)
                vmax = np.nanmax(all_intensity)
            
            # Create a fixed colorbar position
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(temp_ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            
            for i, freq_val in enumerate(self.kgrid_freqs):
                if i % 5 == 0:  # Update progress every 5 frames
                    progress.update_message(f"Generating frame {i+1}/{total_frames}", 
                                          f"Frequency: {freq_val:.2f} THz")
                    self.root.update()
                
                temp_ax.clear()
                cax.clear()
                
                # Get intensity at this frequency
                if hasattr(self, 'sed_result') and self.sed_result.is_complex:
                    intensity = np.sum(np.abs(self.kgrid_sed_data[i, :, :])**2, axis=-1)
                else:
                    if self.kgrid_sed_data.ndim == 3:
                        intensity = np.sum(self.kgrid_sed_data[i, :, :], axis=-1)
                    else:
                        intensity = self.kgrid_sed_data[i, :]
                
                # Apply intensity scaling
                scale_type = self.intensity_scale_var.get()
                if scale_type == "log":
                    intensity = np.log10(np.maximum(intensity, 1e-12))
                elif scale_type == "sqrt":
                    intensity = np.sqrt(np.maximum(intensity, 0))
                elif scale_type == "dsqrt":
                    intensity = np.sqrt(np.sqrt(np.maximum(intensity, 0)))
                
                # Reshape for plotting
                n_kx = len(self.kgrid_kx)
                n_ky = len(self.kgrid_ky)
                intensity_grid = intensity.reshape(n_kx, n_ky).T
                X, Y = np.meshgrid(self.kgrid_kx, self.kgrid_ky)
                
                # Plot with fixed colorbar
                pcm = temp_ax.pcolormesh(X, Y, intensity_grid, cmap=self.colormap_var.get(), 
                                        shading='auto', vmin=vmin, vmax=vmax)
                temp_ax.set_xlabel(self.kgrid_xlabel)
                temp_ax.set_ylabel(self.kgrid_ylabel)
                temp_ax.set_title(f"SED @ {freq_val:.2f} THz")
                
                # Add colorbar to fixed position
                cbar = temp_fig.colorbar(pcm, cax=cax)
                cbar.set_label("Intensity")
                
                # Draw the figure
                temp_fig.canvas.draw()
                
                # Use savefig method for consistent image capture
                buf_io = BytesIO()
                temp_fig.savefig(buf_io, format='png', dpi=gif_dpi, bbox_inches=None)  # Use user-specified DPI
                buf_io.seek(0)
                
                # Read with PIL for consistent array format
                try:
                    from PIL import Image
                    img = Image.open(buf_io)
                    # Convert to RGB if needed and ensure consistent size
                    img = img.convert('RGB')
                    frame_array = np.array(img)
                    frames.append(frame_array)
                except ImportError:
                    # Fallback without PIL
                    buf_io.seek(0)
                    import matplotlib.image as mpimg
                    frame_array = mpimg.imread(buf_io, format='png')
                    # Convert to 0-255 range if needed
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    # Ensure RGB (remove alpha if present)
                    if frame_array.shape[-1] == 4:
                        frame_array = frame_array[:, :, :3]
                    frames.append(frame_array)
                
                buf_io.close()
            
            progress.update_message("Saving GIF...", "Please wait...")
            
            # Verify all frames have the same shape
            if frames:
                first_shape = frames[0].shape
                for i, frame in enumerate(frames):
                    if frame.shape != first_shape:
                        logger.warning(f"Frame {i} has different shape: {frame.shape} vs {first_shape}")
                        # Resize to match first frame if needed
                        if len(frame.shape) == 3 and len(first_shape) == 3:
                            from PIL import Image
                            frame_img = Image.fromarray(frame)
                            frame_img = frame_img.resize((first_shape[1], first_shape[0]), Image.Resampling.LANCZOS)
                            frames[i] = np.array(frame_img)
            
            # Save as GIF with user-specified frame rate
            fps = self.gif_fps_var.get()
            duration = int(1000 / fps)  # Convert fps to ms per frame
            imageio.mimsave(gif_file, frames, duration=duration, loop=0)
            
            progress.close()
            messagebox.showinfo("Save Complete", f"K-grid animation saved as:\n{gif_file.name}")
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            logger.error(f"Error saving k-grid GIF: {e}")
            messagebox.showerror("Error", f"Failed to save k-grid animation:\n{str(e)}")

    def _save_ised_trajectory(self):
        """Save the current iSED trajectory to the output directory"""
        try:
            if not hasattr(self, 'ised_result_path') or not self.ised_result_path:
                messagebox.showerror("Error", "No iSED trajectory available to save")
                return
                
            if not Path(self.ised_result_path).exists():
                messagebox.showerror("Error", "iSED trajectory file not found")
                return
            
            # Create output directory
            output_dir = Path(self.output_dir_var.get())
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use custom filename if provided, otherwise generate one
            custom_name = self.custom_ised_filename_var.get().strip()
            if custom_name:
                # Remove extension if user provided one
                custom_name = Path(custom_name).stem
                filename = f"{custom_name}.dump"
                metadata_filename = f"{custom_name}_metadata.txt"
            else:
                # Generate filename with timestamp and selected point info
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                k_val = getattr(self, 'selected_k', 0)
                w_val = getattr(self, 'selected_w', 0)
                filename = f"ised_trajectory_k{k_val:.3f}_w{w_val:.3f}_{timestamp}.dump"
                metadata_filename = f"ised_metadata_k{k_val:.3f}_w{w_val:.3f}_{timestamp}.txt"
            
            output_file = output_dir / filename
            
            # Copy the file
            import shutil
            shutil.copy2(self.ised_result_path, output_file)
            
            # Save metadata as well
            metadata_file = output_dir / metadata_filename
            with open(metadata_file, 'w') as f:
                f.write(f"iSED Trajectory Metadata\n")
                f.write(f"========================\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Selected k-point: {k_val:.6f}\n")
                f.write(f"Selected frequency: {w_val:.6f} THz\n")
                f.write(f"Number of frames: {self.n_frames_var.get()}\n")
                f.write(f"Rescaling factor: {self.rescale_var.get()}\n")
                f.write(f"K-path direction: {self.direction_var.get()}\n")
                f.write(f"BZ coverage: {self.bz_coverage_var.get()}\n")
                f.write(f"Basis atom types: {self.basis_types_var.get() or 'All'}\n")
                f.write(f"Trajectory file: {self.trajectory_var.get()}\n")
            
            messagebox.showinfo("Save Complete", 
                               f"iSED trajectory saved as:\n{filename}\n\n"
                               f"Metadata saved as:\n{metadata_file.name}")
            
        except Exception as e:
            logger.error(f"Error saving iSED trajectory: {e}")
            messagebox.showerror("Error", f"Failed to save iSED trajectory:\n{str(e)}")

    def _save_current_plot(self):
        """Save the current plot as an image"""
        try:
            if not hasattr(self, 'sed_result') or not self.sed_result:
                messagebox.showerror("Error", "No plot data available to save")
                return
            
            # Create output directory
            output_dir = Path(self.output_dir_var.get())
            output_dir.mkdir(parents=True, exist_ok=True)
            
            format_type = self.image_format_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if hasattr(self, 'sed_calculation_type'):
                calc_type = self.sed_calculation_type.lower().replace("-", "_")
            else:
                calc_type = "unknown"
            
            # Use custom filename if provided, otherwise generate one
            custom_name = self.custom_plot_filename_var.get().strip()
            if custom_name:
                # Remove extension if user provided one
                custom_name = Path(custom_name).stem
                base_filename = custom_name
            else:
                base_filename = f"psa_{calc_type}_plot_{timestamp}"
            
            # Get user-specified parameters
            dpi_val = self.plot_dpi_var.get() if format_type in ['png', 'jpg'] else None
            bbox_inches = None  # Always use default matplotlib bbox
            
            # Apply aspect ratio if specified
            aspect_ratio = self.aspect_ratio_var.get().strip()
            if aspect_ratio and aspect_ratio.lower() != "auto":
                if aspect_ratio.lower() == "equal" or aspect_ratio == "1:1":
                    self.sed_ax.set_aspect('equal', adjustable='box')
                elif ":" in aspect_ratio:
                    # Handle ratios like "4:3", "16:9"
                    try:
                        parts = aspect_ratio.split(":")
                        if len(parts) == 2:
                            width_ratio = float(parts[0])
                            height_ratio = float(parts[1])
                            # Calculate current data range and set aspect
                            xlim = self.sed_ax.get_xlim()
                            ylim = self.sed_ax.get_ylim()
                            self.sed_ax.set_aspect(abs((xlim[1]-xlim[0])/(ylim[1]-ylim[0])) * (height_ratio/width_ratio))
                    except (ValueError, ZeroDivisionError):
                        logger.warning(f"Invalid aspect ratio format: {aspect_ratio}")
                elif aspect_ratio.replace(".", "").replace("-", "").isdigit():
                    # Handle numeric aspect ratios like "1.5", "0.75"
                    try:
                        aspect_val = float(aspect_ratio)
                        self.sed_ax.set_aspect(aspect_val)
                    except ValueError:
                        logger.warning(f"Invalid numeric aspect ratio: {aspect_ratio}")
                else:
                    logger.warning(f"Unrecognized aspect ratio format: {aspect_ratio}")
                
                # Redraw with new aspect ratio
                self.sed_canvas.draw()
            
            if format_type == "png":
                image_file = output_dir / f"{base_filename}.png"
                self.sed_fig.savefig(image_file, format='png', dpi=dpi_val, bbox_inches=bbox_inches)
            elif format_type == "jpg":
                image_file = output_dir / f"{base_filename}.jpg"
                self.sed_fig.savefig(image_file, format='jpeg', dpi=dpi_val, bbox_inches=bbox_inches)
            elif format_type == "svg":
                image_file = output_dir / f"{base_filename}.svg"
                self.sed_fig.savefig(image_file, format='svg', bbox_inches=bbox_inches)
            elif format_type == "pdf":
                image_file = output_dir / f"{base_filename}.pdf"
                self.sed_fig.savefig(image_file, format='pdf', bbox_inches=bbox_inches)
            else:
                messagebox.showerror("Error", f"Unsupported image format: {format_type}")
                return
            
            messagebox.showinfo("Save Complete", f"Plot saved as:\n{image_file.name}")
            
        except Exception as e:
            logger.error(f"Error saving plot image: {e}")
            messagebox.showerror("Error", f"Failed to save plot image:\n{str(e)}")

    def _get_basis_atom_types(self):
        """Helper method to parse basis atom types from the GUI input"""
        if not self.basis_types_var.get().strip():
            return None
        try:
            return [int(x.strip()) for x in self.basis_types_var.get().split(',')]
        except ValueError:
            return None
    
    def _apply_intensity_scaling(self, data, scale_type):
        """Helper method to apply intensity scaling to data"""
        if scale_type == "log":
            return np.log10(np.maximum(data, 1e-12))
        elif scale_type == "sqrt":
            return np.sqrt(np.maximum(data, 0))
        elif scale_type == "dsqrt":
            return np.sqrt(np.sqrt(np.maximum(data, 0)))
        else:  # linear
            return data

    def _create_labeled_scale(self, parent, label_text, variable, from_val, to_val, command=None, format_str="{:.2f}"):
        """Helper method to create label + scale + value display combinations"""
        ttk.Label(parent, text=label_text).pack(anchor="w")
        
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0,5))
        
        scale = ttk.Scale(frame, from_=from_val, to=to_val, variable=variable, 
                         orient="horizontal", command=command)
        scale.pack(side="left", fill="x", expand=True)
        
        # Create value label
        value_var = tk.StringVar()
        value_var.set(format_str.format(variable.get()))
        variable.trace('w', lambda *args: value_var.set(format_str.format(variable.get())))
        
        width = 5 if ".2f" in format_str else 4
        ttk.Label(frame, textvariable=value_var, width=width).pack(side="right")
        
        return frame, scale

    def _on_phase_colormap_change(self):
        """Update plot if Phase colormap is changed."""
        if not hasattr(self, 'sed_calculation_type'):
            return
            
        if self.sed_calculation_type == "K-Path":
            # For k-path, regenerate the plot
            self._generate_sed_plot()
        elif self.sed_calculation_type == "K-Grid":
            # For k-grid, regenerate the heatmap at current frequency
            if hasattr(self, 'kgrid_freq_slider') and hasattr(self, 'kgrid_freqs'):
                current_freq = self.kgrid_freq_slider.get()
                freq_idx = np.argmin(np.abs(self.kgrid_freqs - current_freq))
                self._plot_kgrid_heatmap(freq_idx)


def main():
    """Main function to run the PSA GUI"""
    root = tk.Tk()
    app = PSAMainWindow(root)
    
    # Setup cleanup for temporary files on exit
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit? This will delete any temporary iSED files."):
            app._cleanup_temp_files()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("GUI closed by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        messagebox.showerror("Error", f"Unexpected error:\n{str(e)}")
    finally:
        root.quit()


if __name__ == "__main__":
    main() 