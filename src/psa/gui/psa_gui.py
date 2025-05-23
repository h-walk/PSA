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

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

from psa import TrajectoryLoader, SEDCalculator, SED, SEDPlotter

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
        control_frame = ttk.Frame(parent, width=400)
        parent.add(control_frame)
        
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
        notebook.add(file_frame, text="Input File")
        
        # Trajectory file selection
        ttk.Label(file_frame, text="Trajectory File:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        traj_frame = ttk.Frame(file_frame)
        traj_frame.pack(fill="x", pady=(0,10))
        
        self.trajectory_var = tk.StringVar()
        trajectory_entry = ttk.Entry(traj_frame, textvariable=self.trajectory_var, state="readonly")
        trajectory_entry.pack(side="left", fill="x", expand=True)
        
        ttk.Button(traj_frame, text="Browse...", 
                  command=self._browse_trajectory).pack(side="right", padx=(5,0))
        
        # File format selection
        ttk.Label(file_frame, text="File Format:").pack(anchor="w", pady=(5,0))
        self.format_var = tk.StringVar(value="lammps")
        format_combo = ttk.Combobox(file_frame, textvariable=self.format_var, 
                                   values=["lammps", "xyz", "auto"], state="readonly")
        format_combo.pack(fill="x", pady=(0,10))
        
        # MD parameters
        ttk.Label(file_frame, text="MD Parameters:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        # Timestep
        ttk.Label(file_frame, text="Timestep (ps):").pack(anchor="w")
        self.dt_var = tk.DoubleVar(value=0.005)
        ttk.Entry(file_frame, textvariable=self.dt_var).pack(fill="x", pady=(0,5))
        
        # System dimensions
        dims_frame = ttk.Frame(file_frame)
        dims_frame.pack(fill="x", pady=(5,10))
        ttk.Label(dims_frame, text="System Dimensions:").pack(anchor="w")
        dim_row = ttk.Frame(dims_frame)
        dim_row.pack(fill="x")
        ttk.Label(dim_row, text="nx:").pack(side="left")
        self.nx_var = tk.IntVar(value=50)
        ttk.Entry(dim_row, textvariable=self.nx_var, width=8).pack(side="left", padx=(2, 10))
        ttk.Label(dim_row, text="ny:").pack(side="left")
        self.ny_var = tk.IntVar(value=50)
        ttk.Entry(dim_row, textvariable=self.ny_var, width=8).pack(side="left", padx=(2, 10))
        ttk.Label(dim_row, text="nz:").pack(side="left")
        self.nz_var = tk.IntVar(value=1)
        ttk.Entry(dim_row, textvariable=self.nz_var, width=8).pack(side="left", padx=(2, 0))
        
        # Load trajectory button
        self.load_button = ttk.Button(file_frame, text="Load Trajectory", 
                                     command=self._load_trajectory, state="disabled")
        self.load_button.pack(fill="x", pady=(20,10))
        
        # Status
        self.status_var = tk.StringVar(value="No trajectory loaded")
        ttk.Label(file_frame, textvariable=self.status_var, foreground="blue").pack(anchor="w")
        
    def _create_sed_tab(self, notebook):
        """Create SED calculation parameters tab (now includes k-grid)"""
        sed_frame = ttk.Frame(notebook, padding=(10, 10))
        notebook.add(sed_frame, text="Calculation Parameters")
        
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
        
        # Chirality options (only for K-Path) - moved to be direct child of sed_frame
        self.chirality_frame = ttk.LabelFrame(sed_frame, text="Chirality Options")
        self.chiral_sed_var = tk.BooleanVar(value=False)
        chiral_check = ttk.Checkbutton(self.chirality_frame, text="Calculate Chirality", variable=self.chiral_sed_var, command=self._toggle_chiral_options)
        chiral_check.pack(anchor="w", pady=(5,0))
        ToolTip(chiral_check, text="Enable to calculate chirality (requires coherent summation).")

        self.chiral_options_frame = ttk.Frame(self.chirality_frame)
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
        notebook.add(viz_frame, text="Plot Parameters")

        # Max frequency (shared for both K-Path and K-Grid) - FIRST
        ttk.Label(viz_frame, text="Max Frequency (THz, empty for auto):").pack(anchor="w", pady=(10,0))
        self.max_freq_var = tk.StringVar()
        max_freq_entry = ttk.Entry(viz_frame, textvariable=self.max_freq_var)
        max_freq_entry.pack(fill="x", pady=(0,10))
        max_freq_entry.bind('<Return>', lambda event: self._on_max_freq_change())
        max_freq_entry.bind('<FocusOut>', lambda event: self._on_max_freq_change())
        ToolTip(max_freq_entry, text="Maximum frequency (THz) to display on plots.\nLeave empty for automatic scaling based on data.")

        # Plot Chiral toggle button (only shown when chiral data is available) - SECOND
        self.plot_chiral_frame = ttk.Frame(viz_frame)
        # Don't pack initially - will be shown by _update_viz_controls_state when chiral data is available
        self.plot_chiral_var = tk.BooleanVar(value=False)
        self.plot_chiral_button = ttk.Checkbutton(self.plot_chiral_frame, text="Plot Chiral", 
                                                 variable=self.plot_chiral_var, 
                                                 command=self._on_plot_chiral_toggle)
        self.plot_chiral_button.pack(anchor="w", pady=(0,10))
        ToolTip(self.plot_chiral_button, text="Toggle to display chirality instead of SED intensity.")

        # Intensity scaling (shown for both modes) - THIRD
        ttk.Label(viz_frame, text="Intensity Scaling:").pack(anchor="w")
        self.intensity_scale_var = tk.StringVar(value="dsqrt")
        scale_combo = ttk.Combobox(viz_frame, textvariable=self.intensity_scale_var,
                                  values=["linear", "log", "sqrt", "dsqrt"], state="readonly")
        scale_combo.pack(fill="x", pady=(0,10))
        self.intensity_scale_combo = scale_combo
        self.intensity_scale_combo.bind("<<ComboboxSelected>>", lambda event: self._on_intensity_scale_change())

        # Intensity Colormap (adapt based on mode) - FOURTH
        ttk.Label(viz_frame, text="Intensity Colormap:").pack(anchor="w")
        self.colormap_var = tk.StringVar(value="inferno")
        cmap_combo = ttk.Combobox(viz_frame, textvariable=self.colormap_var,
                                 values=["inferno", "viridis", "plasma", "magma", "hot", "jet"], 
                                 state="readonly")
        cmap_combo.pack(fill="x", pady=(0,10))
        ToolTip(cmap_combo, text="Colormap for SED intensity plot.")
        self.intensity_colormap_combo = cmap_combo
        self.intensity_colormap_combo.bind("<<ComboboxSelected>>", lambda event: self._on_intensity_colormap_change())

        # Phase Colormap (only shown when chiral toggle is enabled) - FIFTH
        self.phase_colormap_frame = ttk.Frame(viz_frame)
        # Don't pack initially - will be shown when chiral toggle is enabled
        ttk.Label(self.phase_colormap_frame, text="Phase Colormap:").pack(anchor="w")
        self.phase_colormap_var = tk.StringVar(value="coolwarm")
        diverging_cmaps = ['coolwarm', 'hsv', 'bwr', 'RdBu', 'RdGy', 'PiYG', 'seismic', 'twilight', 'twilight_shifted']
        self.phase_colormap_combo = ttk.Combobox(self.phase_colormap_frame, textvariable=self.phase_colormap_var,
                                             values=diverging_cmaps, state="readonly")
        self.phase_colormap_combo.pack(fill="x", pady=(0,10))
        ToolTip(self.phase_colormap_combo, text="Colormap for chirality plot (diverging/circular preferred).")
        self.phase_colormap_combo.bind("<<ComboboxSelected>>", lambda event: self._generate_sed_plot())

        # Global Intensity Scaling (only for K-Grid)
        self.global_scale_frame = ttk.Frame(viz_frame)
        # Don't pack initially - will be shown by _update_viz_controls_state
        self.kgrid_global_scale_var = tk.BooleanVar(value=False)
        global_scale_check = ttk.Checkbutton(self.global_scale_frame, text="Global Intensity Scaling", 
                                           variable=self.kgrid_global_scale_var, 
                                           command=lambda: self._plot_kgrid_heatmap(self.kgrid_freq_slider.get()) if hasattr(self, 'kgrid_freq_slider') else None)
        global_scale_check.pack(anchor="w", pady=(0,10))
        
        # Spacer to push Generate Plot button to bottom
        self.viz_spacer_frame = ttk.Frame(viz_frame)
        self.viz_spacer_frame.pack(fill="both", expand=True)

        # Plot status
        self.plot_status_var = tk.StringVar(value="No plot generated")
        ttk.Label(viz_frame, textvariable=self.plot_status_var, foreground="blue").pack(anchor="w", side="bottom")

        # Generate plot button (moved to bottom for better UX)
        self.plot_button = ttk.Button(viz_frame, text="Generate Plot", 
                                     command=self._generate_plot, state="disabled")
        self.plot_button.pack(fill="x", pady=(20,10), side="bottom")

    def _create_ised_tab(self, notebook):
        """Create iSED reconstruction parameters tab"""
        ised_frame = ttk.Frame(notebook)
        notebook.add(ised_frame, text="Atomic Visualization")
        
        ttk.Label(ised_frame, text="Click Parameters:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        # Selected point display
        self.selected_k_var = tk.StringVar(value="k = Not selected")
        self.selected_w_var = tk.StringVar(value="ω = Not selected")
        
        ttk.Label(ised_frame, textvariable=self.selected_k_var, foreground="red").pack(anchor="w")
        ttk.Label(ised_frame, textvariable=self.selected_w_var, foreground="red").pack(anchor="w", pady=(0,10))
        
        # Reconstruction parameters
        ttk.Label(ised_frame, text="Reconstruction Parameters:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,5))
        
        # Number of frames
        ttk.Label(ised_frame, text="Animation Frames:").pack(anchor="w")
        self.n_frames_var = tk.IntVar(value=100)
        ttk.Entry(ised_frame, textvariable=self.n_frames_var).pack(fill="x", pady=(0,5))
        
        # Rescaling factor
        ttk.Label(ised_frame, text="Rescaling Factor:").pack(anchor="w")
        self.rescale_var = tk.StringVar(value="auto")
        rescale_entry = ttk.Entry(ised_frame, textvariable=self.rescale_var)
        rescale_entry.pack(fill="x", pady=(0,10))
        ToolTip(rescale_entry, text="Rescaling factor for iSED atomic displacements.\n'auto' for automatic scaling, or a numerical value (e.g., 0.1, 1.0, 10.0).")
        
        # 3D Visualization Controls
        ttk.Label(ised_frame, text="3D Visualization Controls:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(20,5))
        
        # Atom sizes for different types
        ttk.Label(ised_frame, text="Atom Type 1 Size:").pack(anchor="w")
        self.atom_size_type1_var = tk.DoubleVar(value=0.3)
        size1_frame = ttk.Frame(ised_frame)
        size1_frame.pack(fill="x", pady=(0,5))
        ttk.Scale(size1_frame, from_=0.05, to=2.0, variable=self.atom_size_type1_var, 
                 orient="horizontal", command=self._update_atom_size).pack(side="left", fill="x", expand=True)
        size1_label_var = tk.StringVar()
        size1_label_var.set(f"{self.atom_size_type1_var.get():.2f}")
        self.atom_size_type1_var.trace('w', lambda *args: size1_label_var.set(f"{self.atom_size_type1_var.get():.2f}"))
        ttk.Label(size1_frame, textvariable=size1_label_var, width=5).pack(side="right")
        
        ttk.Label(ised_frame, text="Atom Type 2 Size:").pack(anchor="w")
        self.atom_size_type2_var = tk.DoubleVar(value=0.3)
        size2_frame = ttk.Frame(ised_frame)
        size2_frame.pack(fill="x", pady=(0,5))
        ttk.Scale(size2_frame, from_=0.05, to=2.0, variable=self.atom_size_type2_var, 
                 orient="horizontal", command=self._update_atom_size).pack(side="left", fill="x", expand=True)
        size2_label_var = tk.StringVar()
        size2_label_var.set(f"{self.atom_size_type2_var.get():.2f}")
        self.atom_size_type2_var.trace('w', lambda *args: size2_label_var.set(f"{self.atom_size_type2_var.get():.2f}"))
        ttk.Label(size2_frame, textvariable=size2_label_var, width=5).pack(side="right")
        
        # Atom transparency
        ttk.Label(ised_frame, text="Transparency:").pack(anchor="w")
        self.atom_alpha_var = tk.DoubleVar(value=0.8)
        alpha_frame = ttk.Frame(ised_frame)
        alpha_frame.pack(fill="x", pady=(0,10))
        ttk.Scale(alpha_frame, from_=0.1, to=1.0, variable=self.atom_alpha_var, 
                 orient="horizontal", command=self._update_atom_alpha).pack(side="left", fill="x", expand=True)
        alpha_label_var = tk.StringVar()
        alpha_label_var.set(f"{self.atom_alpha_var.get():.1f}")
        self.atom_alpha_var.trace('w', lambda *args: alpha_label_var.set(f"{self.atom_alpha_var.get():.1f}"))
        ttk.Label(alpha_frame, textvariable=alpha_label_var, width=4).pack(side="right")
        
        # Save permanently checkbox
        self.save_permanently_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ised_frame, text="Save reconstruction file permanently", 
                       variable=self.save_permanently_var,
                       command=self._toggle_save_mode).pack(anchor="w", pady=(5,10))
        
        # Output directory selection (initially hidden)
        self.output_frame = ttk.Frame(ised_frame)
        
        ttk.Label(self.output_frame, text="Output Directory:").pack(anchor="w")
        output_dir_frame = ttk.Frame(self.output_frame)
        output_dir_frame.pack(fill="x", pady=(0,5))
        
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "ised_results"))
        self.output_entry = ttk.Entry(output_dir_frame, textvariable=self.output_dir_var)
        self.output_entry.pack(side="left", fill="x", expand=True)
        
        ttk.Button(output_dir_frame, text="Browse...", 
                  command=self._browse_output_dir).pack(side="right", padx=(5,0))
        
        # Output filename
        ttk.Label(self.output_frame, text="Output Filename:").pack(anchor="w")
        self.output_filename_var = tk.StringVar(value="ised_motion.dump")
        ttk.Entry(self.output_frame, textvariable=self.output_filename_var).pack(fill="x", pady=(0,10))
        
        # Reconstruct button
        self.ised_button = ttk.Button(ised_frame, text="Reconstruct iSED", 
                                     command=self._reconstruct_ised, state="disabled")
        self.ised_button.pack(fill="x", pady=(20,10))
        
        # Animation speed
        ttk.Label(ised_frame, text="Animation Speed (fps):").pack(anchor="w")
        self.anim_fps_var = tk.DoubleVar(value=10.0)
        speed_frame = ttk.Frame(ised_frame)
        speed_frame.pack(fill="x", pady=(0,5))
        ttk.Scale(speed_frame, from_=1.0, to=30.0, variable=self.anim_fps_var, 
                 orient="horizontal", command=self._update_anim_speed).pack(side="left", fill="x", expand=True)
        fps_label_var = tk.StringVar()
        fps_label_var.set(f"{self.anim_fps_var.get():.1f}")
        self.anim_fps_var.trace('w', lambda *args: fps_label_var.set(f"{self.anim_fps_var.get():.1f}"))
        ttk.Label(speed_frame, textvariable=fps_label_var, width=4).pack(side="right")
        
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
            
        def load_worker():
            try:
                # Use thread-safe GUI updates
                self.root.after(0, lambda: self.status_var.set("Loading trajectory..."))
                
                # Load trajectory
                loader = TrajectoryLoader(
                    filename=self.trajectory_file,
                    dt=self.dt_var.get(),
                    file_format=self.format_var.get()
                )
                trajectory = loader.load()
                
                # Initialize SED calculator
                self.sed_calculator = SEDCalculator(
                    traj=trajectory,
                    nx=self.nx_var.get(),
                    ny=self.ny_var.get(),
                    nz=self.nz_var.get()
                )
                
                # Thread-safe GUI updates
                self.root.after(0, lambda: self.status_var.set(f"Trajectory loaded: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms"))
                self.root.after(0, lambda: self.calc_sed_button.config(state="normal"))
                
            except Exception as e:
                logger.error(f"Error loading trajectory: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load trajectory:\n{str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Error loading trajectory"))
                
        # Run in thread to prevent GUI freezing
        thread = threading.Thread(target=load_worker)
        thread.daemon = True
        thread.start()
        
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
                
                basis_types = None
                if self.basis_types_var.get().strip():
                    try:
                        basis_types = [int(x.strip()) for x in self.basis_types_var.get().split(',')]
                    except ValueError:
                        pass
                
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
            if hasattr(self, 'sed_colorbar') and self.sed_colorbar is not None:
                try:
                    self.sed_colorbar.remove()
                    del self.sed_colorbar
                except Exception as e:
                    logger.debug(f"Error removing colorbar: {e}")
                    
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
                scale_type = self.intensity_scale_var.get()
                if scale_type == "log":
                    data_to_plot = np.log10(np.maximum(data_to_plot, 1e-12))
                elif scale_type == "sqrt":
                    data_to_plot = np.sqrt(np.maximum(data_to_plot, 0))
                elif scale_type == "dsqrt":
                    data_to_plot = np.sqrt(np.sqrt(np.maximum(data_to_plot, 0)))

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
            try:
                if hasattr(self, 'sed_colorbar') and self.sed_colorbar:
                    self.sed_colorbar.remove()
                    self.sed_colorbar = None
                if hasattr(self, 'sed_colorbar_ax') and self.sed_colorbar_ax:
                    if self.sed_colorbar_ax.get_figure() == self.sed_fig:
                        self.sed_colorbar_ax.remove()
                    self.sed_colorbar_ax = None
            except Exception as e:
                pass

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
            # Choose output location based on save mode
            if self.save_permanently_var.get():
                # Create output directory for permanent results
                output_dir = Path(self.output_dir_var.get())
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate unique filename if file exists
                base_filename = self.output_filename_var.get()
                if not base_filename.endswith('.dump'):
                    base_filename += '.dump'
                    
                dump_file = output_dir / base_filename
                counter = 1
                while dump_file.exists():
                    name_part = base_filename.replace('.dump', '')
                    dump_file = output_dir / f"{name_part}_{counter:03d}.dump"
                    counter += 1
            else:
                # Use temporary directory
                temp_dir_obj = tempfile.TemporaryDirectory() # Use context manager for auto-cleanup if possible
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
                    
                    if self.save_permanently_var.get():
                        self.ised_status_var.set(f"iSED saved: {dump_file.name}")
                        # Show success message with file location for permanent saves
                        messagebox.showinfo("iSED Complete", 
                                          f"iSED reconstruction saved to:\n{dump_file}\n\n"
                                          f"Selected point: k={self.selected_k:.3f}, ω={self.selected_w:.3f} THz")
                    else:
                        self.ised_status_var.set("iSED complete (temporary)")
                        
                    self.view_motion_button.config(state="normal")
                    self.ised_button.config(state="normal")
                    
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
        
        if calc_type == "K-Path":
            # Hide K-Grid specific controls
            self.global_scale_frame.pack_forget()
            
            # Handle chiral toggle button
            has_phase_data = self.sed_result is not None and hasattr(self.sed_result, 'phase') and self.sed_result.phase is not None
            if has_phase_data:
                # Show the chiral toggle button
                self.plot_chiral_frame.pack(fill="x", pady=(0,10), before=self.viz_spacer_frame)
                # If toggle is currently enabled, show phase colormap
                if self.plot_chiral_var.get():
                    self.phase_colormap_frame.pack(fill="x", pady=(0,10), before=self.viz_spacer_frame)
                else:
                    self.phase_colormap_frame.pack_forget()
            else:
                # Hide chiral controls if no chiral data
                self.plot_chiral_frame.pack_forget()
                self.phase_colormap_frame.pack_forget()
                self.plot_chiral_var.set(False)  # Reset toggle state
                
        else:  # K-Grid
            # Show K-Grid specific controls, hide K-Path specific ones
            self.plot_chiral_frame.pack_forget()
            self.phase_colormap_frame.pack_forget()
            self.plot_chiral_var.set(False)  # Reset toggle state
            self.global_scale_frame.pack(fill="x", pady=(0,10), before=self.viz_spacer_frame)

        # Plot button state depends if any SED result is available
        if hasattr(self, 'sed_result') and self.sed_result:
            self.plot_button.config(state="normal")
        else:
            self.plot_button.config(state="disabled")

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
            self.phase_colormap_frame.pack(fill="x", pady=(0,10), before=self.viz_spacer_frame)
        else:
            # Hide phase colormap when chiral is disabled
            self.phase_colormap_frame.pack_forget()
        
        # Update the plot if we have data
        if hasattr(self, 'sed_calculation_type') and self.sed_calculation_type == "K-Path":
            self._generate_sed_plot()

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
            self.chirality_frame.pack(fill="x", pady=(0,10), before=self.sed_spacer_frame)
        else:  # K-Grid
            self.kpath_frame.pack_forget()
            self.kgrid_frame.pack(fill="x", pady=(0,10), before=self.sed_spacer_frame)
            self.chirality_frame.pack_forget()
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
                
                basis_types = None
                if self.basis_types_var.get().strip():
                    try:
                        basis_types = [int(x.strip()) for x in self.basis_types_var.get().split(',')]
                    except ValueError:
                        pass
                
                sed_obj = self.sed_calculator.calculate(
                    k_points_mags=k_mags,
                    k_vectors_3d=k_vecs,
                    k_grid_shape=k_grid_shape,
                    basis_atom_types=basis_types,
                    summation_mode=self.summation_mode_var.get()
                )
                
                # Store the SED result for the plotting
                self.sed_result = SED(
                    sed=sed_obj.sed,
                    freqs=sed_obj.freqs,
                    k_points=sed_obj.k_points,
                    k_vectors=sed_obj.k_vectors,
                    k_grid_shape=sed_obj.k_grid_shape,
                    phase=None,  # No chirality for k-grid
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
            
        if not self.kgrid_sed_result or self.kgrid_freqs is None or self.kgrid_sed_data is None:
            self.sed_ax.set_title("No k-grid data")
            self.sed_canvas.draw()
            return
            
        try:
            # Get intensity at this frequency
            sed = self.kgrid_sed_result
            freq_val = self.kgrid_freqs[int(freq_idx)]
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
                    
            # Apply intensity scaling
            scale_type = self.intensity_scale_var.get()
            if scale_type == "log":
                intensity = np.log10(np.maximum(intensity, 1e-12))
            elif scale_type == "sqrt":
                intensity = np.sqrt(np.maximum(intensity, 0))
            elif scale_type == "dsqrt":
                intensity = np.sqrt(np.sqrt(np.maximum(intensity, 0)))
            # else linear: do nothing
            
            n_kx = len(self.kgrid_kx)
            n_ky = len(self.kgrid_ky)
            intensity_grid = intensity.reshape(n_kx, n_ky).T # Transpose for correct orientation
            X, Y = np.meshgrid(self.kgrid_kx, self.kgrid_ky)
            # Global scaling with caching for performance
            vmin = vmax = None
            if getattr(self, 'kgrid_global_scale_var', None) and self.kgrid_global_scale_var.get():
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
                        if current_scale_type == "log":
                            all_intensity = np.log10(np.maximum(all_intensity, 1e-12))
                        elif current_scale_type == "sqrt":
                            all_intensity = np.sqrt(np.maximum(all_intensity, 0))
                        elif current_scale_type == "dsqrt":
                            all_intensity = np.sqrt(np.sqrt(np.maximum(all_intensity, 0)))
                        
                        # Cache the results
                        self._cached_global_vmin = vmin = np.nanmin(all_intensity)
                        self._cached_global_vmax = vmax = np.nanmax(all_intensity)
                        self._cached_scale_type = current_scale_type
            
            pcm = self.sed_ax.pcolormesh(X, Y, intensity_grid, cmap=self.colormap_var.get(), shading='auto', vmin=vmin, vmax=vmax)
            self.sed_ax.set_xlabel(self.kgrid_xlabel)
            self.sed_ax.set_ylabel(self.kgrid_ylabel)
            self.sed_ax.set_title(f"SED @ {freq_val:.2f} THz")
            
            # Add colorbar using make_axes_locatable
            divider = make_axes_locatable(self.sed_ax)
            self.sed_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
            self.sed_colorbar_ax.clear()
            self.sed_colorbar = self.sed_fig.colorbar(pcm, cax=self.sed_colorbar_ax)
            self.sed_colorbar.set_label("Intensity", fontsize=12)
            
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