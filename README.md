# Phonon Spectral Analysis (PSA)

A comprehensive Python package for analyzing molecular dynamics trajectories using advanced Phonon Spectral Analysis (PSA) techniques, featuring an interactive GUI, chirality analysis, and inverse SED (iSED) reconstruction.

## 🚀 **New: Interactive GUI Application**

PSA now includes a modern graphical user interface with click-to-select functionality, real-time visualization, and automated iSED reconstruction! 

**Launch with:** `psa-gui`

![PSA GUI Demo](docs/images/psa_gui_screenshot.png)

## Overview

PSA provides cutting-edge tools for analyzing vibrational properties of materials from molecular dynamics simulations:

### **🎯 Core Capabilities**
- **Advanced SED Analysis**: Calculate phonon dispersion relations with multiple calculation modes
- **Interactive GUI**: Modern interface with click-to-select (k,ω) points and real-time plotting
- **Chirality Analysis**: Study chiral phonon modes with phase visualization
- **iSED Reconstruction**: Automatically visualize atomic motion for any selected phonon mode
- **K-Grid Heatmaps**: Frequency-resolved reciprocal space analysis with interactive sliders
- **Professional Visualization**: Publication-ready plots with customizable colormaps and scaling

### **🔬 Advanced Features**
- **Performance Optimization**: Threaded calculations, cached scaling, efficient memory usage
- **Multiple File Formats**: Support for LAMMPS, XYZ, and auto-detection via OVITO
- **Modular Architecture**: Use as standalone package or integrate into existing workflows

## Installation

### Quick Start
```bash
# Clone or download the repository
git clone <repository-url>
cd PSA

# Create virtual environment (recommended)
python3 -m venv psa_env
source psa_env/bin/activate  # Linux/macOS
# psa_env\Scripts\activate    # Windows

# Install in editable mode
pip install -e .

# Launch GUI
psa-gui
```

### Dependencies
**Core Requirements:**
- Python ≥ 3.8
- numpy, matplotlib, tkinter
- tqdm, pyyaml

**Optional but Recommended:**
- ovito (Python bindings for trajectory loading)
- pytest (for testing)


## 🎮 **GUI Usage**

### **Quick Workflow**
1. **Load Trajectory**: Select MD trajectory file with automatic format detection
2. **Choose Mode**: K-Path (dispersion curves) or K-Grid (frequency heatmaps)  
3. **Set Parameters**: Direction, frequency range, grid resolution
4. **Calculate SED**: Background calculation with progress feedback
5. **Interactive Analysis**: Click anywhere on plots to select (k,ω) points
6. **Automatic iSED**: Instant reconstruction and 3D atomic motion visualization

### **Key Interface Features**
- **Click-to-Select**: Click any point on SED plots for automatic iSED reconstruction
- **Dynamic Controls**: Smart parameter organization based on calculation mode
- **Real-time Updates**: Modify parameters and regenerate plots instantly
- **Professional Export**: Save animations, plots, and reconstructed trajectories

See `GUI_README.md` for comprehensive GUI documentation.

## 📚 **Programmatic Usage**

### **Basic SED Analysis**
```python
from psa.core.sed_calculator import SEDCalculator
from psa.io.trajectory_loader import TrajectoryLoader

# Load trajectory
loader = TrajectoryLoader()
traj_data = loader.load_trajectory("trajectory.lammpstrj")

# Initialize calculator
calc = SEDCalculator(
    positions=traj_data['positions'],
    velocities=traj_data['velocities'], 
    masses=traj_data['masses'],
    types=traj_data['types'],
    lattice=traj_data['lattice']
)

# Calculate K-Path SED
sed_result = calc.calculate_kpath_sed(
    direction=[1, 0, 0],        # Miller indices
    bz_coverage=4.0,            # Reciprocal space coverage
    n_k=100,                    # Number of k-points
    basis_atom_types=[1, 2],    # Atom types to include
    summation_mode='coherent'   # Complex or intensity summation
)

# Plot results
from psa.plotting.sed_plotter import SEDPlotter
plotter = SEDPlotter()
plotter.plot_sed_dispersion(sed_result)
```

### **Advanced Chirality Analysis**
```python
# Calculate chiral SED with phase information
chiral_result = calc.calculate_chiral_sed(
    direction=[1, 1, 0],
    bz_coverage=3.0,
    n_k=80,
    chiral_axis='z'  # Chirality calculation axis
)

# Plot chirality with phase colormap
plotter.plot_chiral_dispersion(
    chiral_result,
    intensity_colormap='inferno',
    phase_colormap='coolwarm'
)
```

### **K-Grid Heatmap Analysis**
```python
# Calculate k-grid for detailed frequency analysis
kgrid_result = calc.calculate_kgrid_sed(
    plane='xy',                 # Plane selection
    k_ranges=(-5, 5, -5, 5),   # kx_min, kx_max, ky_min, ky_max
    n_kx=50, n_ky=50,          # Grid resolution
    k_fixed=0.0                # Fixed k_z value
)

# Create frequency heatmap
plotter.plot_kgrid_heatmap(
    kgrid_result,
    frequency_index=100,       # Specific frequency slice
    intensity_scaling='dsqrt'  # Dynamic square root scaling
)
```

### **iSED Reconstruction**
```python
from psa.core.ised_reconstructor import iSEDReconstructor

# Reconstruct atomic motion for selected (k,ω) point
reconstructor = iSEDReconstructor(sed_result)
ised_motion = reconstructor.reconstruct_motion(
    k_target=2.5,              # Target k-value (2π/Å)
    omega_target=25.0,         # Target frequency (THz)
    n_frames=100,              # Animation frames
    rescale_factor=0.2         # Amplitude scaling
)

# Save as LAMMPS trajectory
reconstructor.save_trajectory(
    ised_motion, 
    "ised_motion.dump",
    format='lammps'
)
```

## 🔬 **Advanced Features**

### **Directional Projection Enhancement**
PSA now uses proper reciprocal lattice vector projection:
```python
# Physical k-path coverage using reciprocal lattice projection
# Coverage factor represents true BZ extent in specified direction
sed_result = calc.calculate_kpath_sed(
    direction=[1, 1, 0],
    bz_coverage=2.0  # 2× BZ boundary in [110] direction
)
```

### **Performance Optimizations**
- **Threaded Calculations**: All SED calculations run in background
- **Memory Efficiency**: Smart frequency filtering and data management
- **Cached Scaling**: Global intensity scaling with intelligent caching
- **Optimized Algorithms**: Efficient FFT and correlation calculations

### **File Format Support**
```python
# Automatic format detection
supported_formats = [
    "*.lammpstrj",     # LAMMPS trajectory
    "*.xyz",           # XYZ format  
    "*.dump",          # LAMMPS dump
    "*.*"              # OVITO auto-detection
]
```

## 📁 **Code Examples**

Comprehensive examples demonstrating PSA capabilities:

- **`examples/basic_sed_analysis.py`**: Standard SED calculation and plotting
- **`examples/chiral_sed_analysis.py`**: Chirality analysis with phase visualization  
- **`examples/kgrid_analysis.py`**: K-grid heatmap generation and analysis
- **`examples/ised_reconstruction.py`**: Automated iSED reconstruction workflow
- **`examples/visualization_gallery.py`**: Advanced plotting and customization
- **`examples/batch_processing.py`**: High-throughput analysis workflows

## 🧪 **Testing**

Run the comprehensive test suite:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests with coverage
pytest --cov=psa tests/

# Run specific test categories
pytest tests/test_sed_calculator.py    # Core calculations
pytest tests/test_gui/                 # GUI functionality  
pytest tests/test_visualization.py    # Plotting features
```

## 📖 **Documentation**

### **Available Documentation**
- **`GUI_README.md`**: Comprehensive GUI usage guide
- **`docs/api/`**: Detailed API reference for all modules
- **`examples/`**: Runnable code examples
- **`tests/`**: Test suite demonstrating expected usage

### **Planned Documentation**
- **User Guides**: Common analysis workflows and best practices
- **Tutorials**: Step-by-step analysis examples  
- **API Reference**: Complete class and method documentation
- **Performance Guide**: Optimization tips for large systems

## 🆕 **Recent Improvements**

### **Version 2.0 Features**
- **🎮 Interactive GUI**: Modern interface with click-to-select functionality
- **⚡ Performance**: 5-10× faster calculations with optimized algorithms
- **📊 K-Grid Analysis**: Real-time frequency heatmaps with sliders

### **Interface Improvements**
- **Unified Workflow**: Streamlined parameter organization
- **Smart Controls**: Dynamic UI based on calculation mode
- **Real-time Feedback**: Instant parameter validation and updates
- **Professional Export**: High-quality plot and animation output

## 🤝 **Contributing**

We welcome contributions! Please see our contributing guidelines:

### **Development Setup**
```bash
# Clone repository
git clone <repository-url>
cd PSA

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests before committing
pytest tests/
```

### **Contribution Areas**
- **New Features**: Additional analysis methods or visualization options
- **Performance**: Algorithm optimizations and memory efficiency
- **Documentation**: Examples, tutorials, and API documentation
- **Testing**: Unit tests and integration test coverage
- **GUI Enhancements**: New interface features and usability improvements

## 📄 **Citation**

If you use PSA in your research, please cite:

```bibtex
@software{psa_package,
  title = {Phonon Spectral Analysis (PSA): Interactive Analysis of Molecular Dynamics Trajectories},
  author = {[Author Names]},
  year = {2024},
  url = {[Repository URL]},
  version = {2.0}
}
```

## 📋 **License**

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🚀 **Quick Links**

- **🎮 GUI Tutorial**: `GUI_README.md`
- **📚 Examples**: `examples/` directory
- **🧪 Tests**: `pytest tests/`
- **📖 API Docs**: `docs/api/`
- **🐛 Issues**: [GitHub Issues]
- **💬 Discussions**: [GitHub Discussions]

---

**Ready to explore phonon properties?** Launch the GUI with `psa-gui` or dive into the examples! 
