#  SDA

SDA (Spectral Displacement Analysis) is a Python tool for analyzing phonon dynamics in molecular dynamics simulations. 

## Theoretical Background

SDA analyzes the space-time Fourier transform of atomic displacements to decompose the dynamics into contributions from different vibrational modes. The analysis reveals:

* Phonon dispersion relations
* Frequency filtered dynamics
* Spatially distributed vibrational patterns
* Population distribution across modes

The tool works by:

* Computing atomic displacements from average positions
* Performing space-time Fourier transforms
* Filtering specific modes in (k, ω) space
* Reconstructing filtered trajectories


## Getting Started

### Prerequisites
```bash
python >= 3.x
numpy
matplotlib
ovito  # For trajectory handling
tqdm   # For progress bars
pyyaml # For configuration
```

### Installation
```bash
git clone https://github.com/walkavelii/SDA.git
cd SDA
pip install numpy matplotlib ovito tqdm pyyaml
```

### Basic Usage
```bash
python SDA.py path/to/trajectory --config config.yaml
```

## Configuration

Example `config.yaml`:
```yaml
# System parameters
dt: 0.005                   # Timestep (ps)
nx: 60                      # Grid points (x)
ny: 60                      # Grid points (y)
nz: 1                       # Grid points (z)

# Analysis settings
direction: 'x'              # k-path direction ('x','y','z' or [vx,vy,vz])
bz_coverage: 1.0            # Brillouin zone coverage
n_kpoints: 60              # Number of k-points

# Filtering parameters
wmin: 0                     # Min frequency (THz)
wmax: 50                    # Max frequency (THz)
kmin: 0.0                   # Min k-value (2π/Å)
kmax: 2.0                   # Max k-value (2π/Å)
amplitude: 0.5              # Reconstruction amplitude scaling
```

## Output Files

* **Analysis Data**: Cached NumPy arrays for positions, types, timings, and box parameters
* **Visualizations**: Full (sd_global.png) and filtered (sd_filtered.png) spectral density plots
* **Results**: Reconstructed trajectory (reconstructed.lammpstrj) for filtered modes

## Workflow Example

1. Run your MD simulation and save atomic positions
2. Create a config.yaml with your parameters
3. Run SDA:
   ```bash
   python SDA.py dump.lammpstrj --config config.yaml
   ```
4. Analyze results in sd_*.png plots
5. Visualize filtered modes in OVITO using reconstructed.lammpstrj

## Technical Notes

* Uses caching to speed up subsequent analyses
* Add --reload flag to force reprocessing
* Currently supports Cartesian k-vectors only
* Memory usage correlates with system size

## Contributing

We welcome contributions! Please check our issues page or submit a pull request.

## License


## Citation

```bibtex
[Citation format pending]
```
