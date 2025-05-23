# PSA GUI - Interactive Phonon Spectral Analysis

A modern graphical user interface for Phonon Spectral Analysis with interactive SED plotting and atomic motion visualization.

## Features

### 🎯 **Interactive SED Analysis**
- **Click-to-Select**: Click anywhere on the SED dispersion plot to select (k,ω) points
- **Real-time Parameter Adjustment**: Modify calculation parameters and regenerate plots instantly
- **Multiple Visualization Options**: Various colormaps, intensity scaling, and frequency ranges

### 🔬 **Automatic iSED Reconstruction**
- **One-Click Reconstruction**: Automatically reconstruct iSED for any selected point
- **3D Atomic Motion**: View animated atomic displacements in real-time
- **Export Capabilities**: Save atomic motion as LAMMPS dump files

### 📊 **Professional Interface**
- **Tabbed Organization**: Clean separation of input, parameters, visualization, and reconstruction
- **Progress Tracking**: Real-time status updates during calculations
- **Error Handling**: User-friendly error messages and validation

## Installation & Launch

### Quick Start
```bash
# Install the package
pip install -e .

# Launch GUI (Method 1 - Launcher script)
python psa_gui_launcher.py

# Launch GUI (Method 2 - Entry point)
psa-gui

# Launch GUI (Method 3 - Module)
python -m psa.gui.psa_gui
```

### Dependencies
- Python ≥ 3.8
- numpy
- matplotlib
- tkinter (usually included with Python)
- ovito (for trajectory loading)
- pyyaml, tqdm

## Usage Workflow

### 1. **📂 Load Trajectory**
- Go to **"Input Files"** tab
- Click **"Browse..."** to select your trajectory file
- Set **MD parameters** (timestep, system dimensions)
- Click **"Load Trajectory"**

### 2. **⚙️ Configure SED Parameters**
- Go to **"SED Parameters"** tab
- Set **k-path direction** (e.g., `[1,1,0]`, `x`, `y`)
- Adjust **number of k-points** and **Brillouin zone coverage**
- Specify **lattice parameter** (leave empty for auto-detection)
- Set **basis atom types** if needed
- Click **"Calculate SED"**

### 3. **📊 Generate Interactive Plot**
- Go to **"Visualization"** tab
- Set **plot parameters** (max frequency, colormap, intensity scaling)
- Click **"Generate SED Plot"**
- The plot appears in the **"SED Dispersion"** tab

### 4. **🎯 Interactive Point Selection**
- **Click anywhere** on the SED intensity plot
- Selected (k,ω) coordinates are displayed in the **"iSED Reconstruction"** tab
- The **"Reconstruct iSED"** button becomes enabled

### 5. **🔬 View Atomic Motion**
- Go to **"iSED Reconstruction"** tab
- Adjust **reconstruction parameters** if needed
- Click **"Reconstruct iSED"**
- Switch to **"iSED & Atomic Motion"** tab to see:
  - **iSED dispersion plot** (left panel)
  - **Animated 3D atomic motion** (right panel)

## Interface Overview

### Left Panel (Controls)
```
📁 Input Files
├── Trajectory file selection
├── File format options
├── MD parameters (timestep, dimensions)
└── Load trajectory

⚙️ SED Parameters  
├── K-path direction
├── Number of k-points
├── Brillouin zone coverage
├── Lattice parameter
├── Basis atom selection
└── Calculate SED

📊 Visualization
├── Max frequency
├── Intensity scaling
├── Colormap selection
└── Generate plot

🔬 iSED Reconstruction
├── Selected point display
├── Animation parameters
├── Reconstruct iSED
└── View in external software
```

### Right Panel (Visualization)
```
📊 SED Dispersion
├── Interactive intensity plot
├── Click-to-select functionality
├── Zoom/pan toolbar
└── Colorbar

🎬 iSED & Atomic Motion
├── iSED reconstruction plot
├── 3D animated atomic motion
└── Frame-by-frame visualization
```

## Key Interactive Features

### 🖱️ **Click Detection**
- Click anywhere on the SED plot to select (k,ω) coordinates
- Coordinates are automatically detected and displayed
- iSED reconstruction becomes available immediately

### 🎬 **Motion Animation**
- Automatic 3D animation of reconstructed atomic motion
- Real-time frame updates showing phonon mode dynamics
- Consistent axis scaling for clear visualization

### 💾 **Export Options**
- LAMMPS dump files for external visualization
- Automatic file generation in temporary directories
- Integration with OVITO (if available)

## Tips for Best Results

1. **File Formats**: LAMMPS trajectory files (`.lammpstrj`) work best
2. **System Dimensions**: Set `nx`, `ny`, `nz` to match your simulation cell
3. **k-path Directions**: Use standard crystallographic directions like `[1,1,0]`, `[1,0,0]`
4. **Intensity Scaling**: Try `dsqrt` for best visualization of weak features
5. **Frequency Range**: Set max frequency to focus on phonon bands of interest

## Troubleshooting

### Common Issues
- **Import Errors**: Make sure PSA package is installed (`pip install -e .`)
- **OVITO Not Found**: Install OVITO separately for external motion viewing
- **Large Files**: GUI may be slow with very large trajectory files
- **Memory Issues**: Reduce number of k-points or frames for large systems

### Performance Tips
- Use smaller numbers of k-points for initial exploration
- Reduce animation frames for faster iSED reconstruction
- Close unused plot windows to free memory

## Advanced Features

### **Threading**
- All calculations run in background threads
- GUI remains responsive during long computations
- Progress updates show calculation status

### **Error Handling**
- Comprehensive error checking and user feedback
- Graceful handling of invalid inputs
- Detailed logging for debugging

### **Extensibility**
- Modular design allows easy feature additions
- Clean separation between GUI and calculation logic
- Standard matplotlib integration for custom plotting

---

**Need Help?** Check the main README.md for more details about the PSA package and its capabilities. 