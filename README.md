# Phonon Spectral Analysis (PSA)

A Python package for analyzing molecular dynamics trajectories using Phonon Spectral Analysis (PSA) and Inverse SED (iSED) reconstruction.

## Overview

PSA is a powerful tool for analyzing vibrational properties of materials from molecular dynamics simulations. This package provides:

- SED for calculating phonon dispersion relations.
- Chiral SED analysis for studying chiral phonon modes.
- Inverse SED (iSED) reconstruction for visualizing phonon modes.
- Comprehensive visualization tools for 2D and 3D plots.
- Support for various trajectory file formats (via OVITO integration).
- A command-line interface (CLI) for easy execution with configuration files.
- A modular structure for programmatic use and extension.

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- Matplotlib
- OVITO (Python bindings, for trajectory loading)
- tqdm
- PyYAML (for CLI configuration file usage)

### Installation Steps

1.  **Navigate to the project root directory.**
    This is the directory containing this `README.md` file and the `src/` directory.

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv psa_env
    source psa_env/bin/activate  # On Linux/macOS
    # psa_env\Scripts\activate    # On Windows
    ```

3.  **Install the package in editable mode:**
    This command installs the `psa` package from the `src` directory, allowing you to make changes to the source code that are immediately reflected.
    ```bash
    pip install -e .
    ```

4.  **Install `pytest` for running tests (optional but recommended for development):**
    ```bash
    pip install pytest
    ```


## Code Examples

Runnable Python script examples demonstrating various features of the `psa` package can be found in the `examples/` directory:

-   `basic_sed_analysis.py`: Demonstrates standard SED calculation and plotting.
-   `chiral_sed_analysis.py`: Shows how to perform chiral SED.
-   `ised_reconstruction.py`: Illustrates iSED mode reconstruction.
-   `visualization_example.py`: Shows various plotting capabilities.


## Running Tests

To run the test suite, ensure you have `pytest` installed (`pip install pytest`). Then, from the project root directory, simply run:

```bash
pytest
```

## Documentation Structure

The full documentation is a work in progress and can be found in the `docs/` directory:

-   `docs/api/`: Detailed API reference for modules and classes.
-   `docs/examples/`: (Planned) Example scripts and notebooks. (Note: current runnable examples are in `examples/`)
-   `docs/guides/`: (Planned) User guides for common tasks.
-   `docs/tutorials/`: (Planned) Step-by-step tutorials.

## Features

-   **Trajectory Analysis:**
    -   Support for may trajectory formats (via OVITO Python library).
-   **SED Calculation**
    -   Regular SED analysis (intensity)
    -   Chiral SED analysis (phase difference)
    -   Customizable k-point sampling and path definitions
    -   Support for different polarization directions (implicitly x,y,z)
-   **Visualization**
    -   2D intensity and phase plots
    -   3D intensity and phase dispersion plots (when multiple k-directions are analyzed)
    -   Customizable plot styles and output
-   **iSED Reconstruction**
    -   Visualization of atomic motion for specific (k,ω) modes
    -   Output of reconstructed motion as LAMMPS dump files
    -   Automatic or manual rescaling of reconstructed amplitudes

## Contributing

Contributions are welcome! Please outline your proposed changes in an issue or pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if one is added). 
