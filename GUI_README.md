# PSA GUI - Interactive Phonon Spectral Analysis

A modern graphical user interface for Phonon Spectral Analysis with interactive SED plotting, k-grid heatmaps, chirality analysis, and atomic motion visualization.

## Features

### 🎯 **Interactive SED Analysis**
- **Click-to-Select**: Click anywhere on the SED dispersion plot to select (k,ω) points
- **Dual Calculation Modes**: K-Path dispersion curves and K-Grid frequency heatmaps
- **Real-time Parameter Adjustment**: Modify visualization parameters and regenerate plots instantly
- **Advanced Colormaps**: Multiple intensity and phase colormaps with smart defaults

### 🔬 **Automatic iSED Reconstruction**
- **One-Click Reconstruction**: Automatically reconstruct iSED for any selected point
- **3D Atomic Motion**: View animated atomic displacements with customizable visualization
- **Smart Export**: Temporary or permanent file saving with external viewer integration

### 🧮 **Chirality Analysis**
- **Interactive Chirality Toggle**: Switch between SED intensity and chirality plots
- **Dynamic Phase Colormaps**: Specialized colormaps for phase visualization
- **Multiple Chiral Axes**: X, Y, or Z axis chirality calculation

### 📊 **Professional Interface**
- **Unified Tab Organization**: Streamlined workflow with logical parameter grouping
- **Performance Optimizations**: Cached global scaling, optimized frequency filtering
- **Enhanced UX**: Buttons positioned at bottom, visual separators, contextual tooltips

## Installation & Launch

### Quick Start
```bash
# Install the package
pip install -e .

# Launch GUI (Recommended)
psa-gui

# Alternative methods
python psa_gui_launcher.py
python -m psa.gui.psa_gui
```

### Dependencies
- Python ≥ 3.8
- numpy, matplotlib, tkinter
- PSA package dependencies (tqdm, pyyaml, etc.)
- ovito (optional, for external trajectory visualization)

## Updated Workflow

### 1. **📂 Input File Setup**
- **"Input File"** tab for trajectory loading
- Support for LAMMPS, XYZ, and auto-detection
- Configurable MD parameters (timestep, system dimensions)
- **Load Trajectory** with progress feedback

### 2. **⚙️ Calculation Parameters**
- **"Calculation Parameters"** tab with dual modes:

#### **K-Path Mode** (Dispersion Curves)
- **K-path Direction**: Supports `[h,k,l]` Miller indices, named directions (`x`, `y`, etc.)
- **Reciprocal Space Coverage**: 🆕 **Improved directional projection** using reciprocal lattice vectors
- **Number of k-points**: Validated integer input
- **Chirality Options**: Optional chirality calculation with axis selection

#### **K-Grid Mode** (Frequency Heatmaps)  
- **Plane Selection**: xy, yz, or zx planes with dynamic axis labeling
- **K-space Ranges**: Independent min/max for each axis direction
- **Grid Resolution**: Configurable n_kx, n_ky values
- **Fixed Axis Value**: Perpendicular k-component setting

#### **Common Parameters**
- **Basis Atom Types**: Comma-separated list or empty for all atoms
- **Summation Mode**: Coherent (complex) or incoherent (intensity) summation

### 3. **📊 Plot Parameters & Visualization**
- **"Plot Parameters"** tab with smart parameter ordering:

#### **Shared Parameters**
1. **Max Frequency**: Controls frequency range for both K-Path and K-Grid plots
2. **Intensity Scaling**: Linear, log, sqrt, dsqrt options
3. **Intensity Colormap**: Optimized colormap selection

#### **K-Path Specific**
4. **Plot Chiral toggle**: 🆕 **Dynamic toggle** instead of dropdown menu
5. **Phase Colormap**: 🆕 **Appears dynamically** when chiral toggle is enabled

#### **K-Grid Specific**
4. **Global Intensity Scaling**: 🆕 **Performance-optimized** cross-frequency scaling
5. **Frequency Slider**: 🆕 **Real THz values** with proper frequency resolution

### 4. **🎯 Interactive Point Selection & Analysis**
- Click anywhere on SED or chirality plots
- **Real-time coordinate feedback** in "Atomic Visualization" tab
- Green crosshair marker with precise positioning

### 5. **🔬 Enhanced Atomic Visualization**
- **"Atomic Visualization"** tab with comprehensive controls:
  - **Reconstruction Parameters**: Animation frames, rescaling factors
  - **3D Visualization Controls**: Per-atom-type sizing, transparency control
  - **Animation Controls**: Variable FPS, play/pause, view reset
  - **Save Options**: Temporary (auto-cleanup) or permanent file saving

### 6. **🎬 Real-Space & Reciprocal-Space Views**
#### **"Reciprocal Space"** Tab
- **K-Path**: Interactive dispersion plots with click selection
- **K-Grid**: Real-time frequency heatmaps with slider control

#### **"Real Space"** Tab  
- **3D Atomic Motion**: Full-featured animation with navigation toolbar
- **Enhanced Visualization**: Multi-type atom coloring, size controls, transparency
- **Professional Navigation**: 3D zoom, pan, rotate with toolbar integration

## Key Improvements & New Features

### 🔧 **Enhanced Reciprocal Space Coverage**
- **Directional Projection**: Uses proper b₁·k̂, b₂·k̂, b₃·k̂ projections instead of simple |a₁|
- **Physical Meaning**: Coverage factor now represents true BZ extent in specified direction
- **Better Logging**: Detailed projection information in console output

### ⚡ **Performance Optimizations**
- **Cached Global Scaling**: K-Grid intensity scaling with smart caching system
- **Optimized Frequency Filtering**: Efficient max frequency application
- **Threaded Calculations**: All SED calculations run in background threads

### 🎛️ **Improved User Experience**
- **Smart Button Placement**: Calculate/Generate buttons moved to bottom of tabs
- **Visual Separators**: PanedWindow with visible sash for clear panel separation
- **Dynamic Controls**: Chiral toggle with conditional phase colormap appearance
- **Enhanced Tooltips**: Comprehensive parameter explanations

### 📈 **K-Grid Enhancements**
- **Real Frequency Values**: Slider uses actual THz values instead of meaningless indices
- **Proper Frequency Resolution**: Slider increments match SED frequency resolution
- **Improved Titles**: Clean "SED @ X.XX THz" format
- **Max Frequency Integration**: Slider range respects max frequency parameter

### 🧮 **Advanced Chirality Features**
- **Toggle-Based Interface**: Simplified chiral/SED switching with single button
- **Conditional UI Elements**: Phase colormap appears only when needed
- **Multiple Chiral Axes**: X, Y, Z axis support with proper component selection
- **Smart State Management**: Automatic mode detection and control updates

## Interface Layout

### **Left Panel: Control Tabs**
```
📁 Input File
├── Trajectory file browser & format selection
├── MD parameters (dt, nx, ny, nz)
└── Load Trajectory button

⚙️ Calculation Parameters
├── K-Path/K-Grid mode radio buttons
├── K-Path: direction, coverage, k-points, chirality
├── K-Grid: plane, ranges, resolution  
├── Common: basis atoms, summation mode
└── Calculate SED button

📊 Plot Parameters  
├── Max Frequency (shared)
├── Plot Chiral toggle (K-Path only)
├── Intensity Scaling & Colormap
├── Phase Colormap (when chiral enabled)
├── Global Scaling (K-Grid only)
└── Generate Plot button

🔬 Atomic Visualization
├── Selected point coordinates
├── Reconstruction & 3D controls
├── Animation parameters & controls
└── Save options & external viewing
```

### **Right Panel: Visualization Tabs**
```
🔄 Reciprocal Space
├── K-Path: Interactive dispersion plots
├── K-Grid: Frequency heatmaps with slider
├── Click-to-select functionality
└── Navigation toolbar

🎬 Real Space
├── 3D animated atomic motion
├── Multi-type atom visualization
├── Advanced navigation controls
└── Professional 3D toolbar
```

## Technical Improvements

### **Threading & Performance**
- Background SED calculations with progress updates
- GUI responsiveness during long computations  
- Cached scaling computations for k-grid plots
- Efficient frequency filtering and slider updates

### **State Management**
- Smart control state updates based on calculation type
- Dynamic UI element visibility (chiral controls, sliders)
- Proper cleanup of temporary files on exit
- Session state preservation across mode switches

### **Error Handling & Validation**
- Comprehensive input validation with real-time feedback
- Graceful error handling with user-friendly messages
- Detailed logging for debugging and progress tracking
- Smart fallbacks for edge cases

## Tips for Optimal Usage

### **Performance Tips**
1. **K-Grid Calculations**: Start with smaller grids (20×20) for initial exploration
2. **Global Scaling**: Disable for faster individual frequency plots
3. **Max Frequency**: Set appropriate limits to reduce memory usage
4. **Animation Speed**: Adjust FPS based on system performance

### **Analysis Workflow**
1. **Explore with K-Path**: Use k-path mode for initial dispersion analysis
2. **Detailed Analysis with K-Grid**: Switch to k-grid for specific frequency investigation
3. **Chirality Investigation**: Enable chirality for chiral materials analysis
4. **Interactive Selection**: Click interesting features for iSED reconstruction

### **Visualization Best Practices**
1. **Intensity Scaling**: Use `dsqrt` for best contrast in most cases
2. **Frequency Range**: Set max frequency to focus on relevant phonon branches
3. **Colormap Selection**: `inferno` for intensity, `coolwarm` for phase
4. **3D Atom Sizing**: Adjust per-type sizes for clear visualization

## Troubleshooting

### **Common Issues & Solutions**
- **Slow Performance**: Reduce k-points/grid resolution, disable global scaling
- **Memory Issues**: Lower max frequency, reduce animation frames
- **Click Detection**: Ensure plot is fully generated before clicking
- **File Export**: Check write permissions for permanent save locations

### **Advanced Debugging**
- Console output provides detailed calculation progress
- Error messages include specific parameter guidance
- Log level can be adjusted for more detailed debugging
- Thread-safe GUI updates prevent interface freezing

---

**For More Information**: See the main README.md for PSA package details and the full documentation for advanced features and API usage. 