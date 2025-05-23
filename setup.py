from setuptools import setup, find_packages

setup(
    name="psa",
    version="0.1.0",
    description="Phonon Spectral Analysis package with interactive GUI",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "ovito"
    ],
    entry_points={
        'console_scripts': [
            'psa=psa.cli:main',
            'psa-gui=psa.gui.psa_gui:main',
        ],
    },
    python_requires=">=3.8",
) 