import pytest
import numpy as np
from pathlib import Path
from sda.core.sed import SED

@pytest.fixture
def valid_sed_data():
    """Provides a minimal set of valid data for SED initialization."""
    n_freqs = 10
    n_kpoints = 5
    sed_array = np.random.rand(n_freqs, n_kpoints, 3).astype(np.complex64)
    sed_array += 1j * np.random.rand(n_freqs, n_kpoints, 3).astype(np.complex64)
    freqs = np.linspace(0, 10, n_freqs, dtype=np.float32)
    k_points = np.linspace(0, 1, n_kpoints, dtype=np.float32)
    k_vectors = np.random.rand(n_kpoints, 3).astype(np.float32)
    phase = np.random.rand(n_freqs, n_kpoints).astype(np.float32)
    return {
        "sed": sed_array,
        "freqs": freqs,
        "k_points": k_points,
        "k_vectors": k_vectors,
        "phase": phase,
    }

def test_sed_initialization(valid_sed_data):
    """Test successful SED initialization."""
    sed_obj = SED(**valid_sed_data)
    np.testing.assert_array_equal(sed_obj.sed, valid_sed_data["sed"])
    assert sed_obj.phase is not None

def test_sed_intensity_property(valid_sed_data):
    """Test the intensity property calculation."""
    sed_obj = SED(**valid_sed_data)
    expected_intensity = np.sum(np.abs(valid_sed_data["sed"])**2, axis=-1).astype(np.float32)
    np.testing.assert_allclose(sed_obj.intensity, expected_intensity, atol=1e-6)

def test_sed_intensity_empty():
    """Test intensity calculation with empty SED data."""
    empty_sed = SED(sed=np.array([]).reshape(0,0,3), freqs=np.array([]), k_points=np.array([]), k_vectors=np.array([]).reshape(0,3))
    assert empty_sed.intensity.shape == (0,0) # Or whatever shape is appropriate for empty
    assert empty_sed.intensity.size == 0

def test_sed_save_load(valid_sed_data, tmp_path):
    """Test saving and loading SED data."""
    sed_obj_orig = SED(**valid_sed_data)
    base_path = tmp_path / "test_sed_data"

    sed_obj_orig.save(base_path)

    # Check if files were created
    assert (base_path.with_suffix('.sed.npy')).exists()
    assert (base_path.with_suffix('.freqs.npy')).exists()
    assert (base_path.with_suffix('.k_points.npy')).exists()
    assert (base_path.with_suffix('.k_vectors.npy')).exists()
    assert (base_path.with_suffix('.phase.npy')).exists()

    sed_obj_loaded = SED.load(base_path)

    np.testing.assert_allclose(sed_obj_loaded.sed, sed_obj_orig.sed, atol=1e-6)
    np.testing.assert_allclose(sed_obj_loaded.freqs, sed_obj_orig.freqs, atol=1e-6)
    np.testing.assert_allclose(sed_obj_loaded.k_points, sed_obj_orig.k_points, atol=1e-6)
    np.testing.assert_allclose(sed_obj_loaded.k_vectors, sed_obj_orig.k_vectors, atol=1e-6)
    np.testing.assert_allclose(sed_obj_loaded.phase, sed_obj_orig.phase, atol=1e-6)

def test_sed_save_load_no_phase(valid_sed_data, tmp_path):
    """Test saving and loading SED data when phase is None."""
    data_no_phase = valid_sed_data.copy()
    data_no_phase["phase"] = None
    sed_obj_orig = SED(**data_no_phase)
    base_path = tmp_path / "test_sed_no_phase"

    sed_obj_orig.save(base_path)
    assert not (base_path.with_suffix('.phase.npy')).exists()

    sed_obj_loaded = SED.load(base_path)
    assert sed_obj_loaded.phase is None
    np.testing.assert_allclose(sed_obj_loaded.sed, sed_obj_orig.sed, atol=1e-6)

def test_sed_load_missing_files(tmp_path):
    """Test SED.load raises FileNotFoundError if essential files are missing."""
    base_path = tmp_path / "test_sed_missing"
    # Create only one of the required files
    np.save(base_path.with_suffix('.sed.npy'), np.array([1])) 
    with pytest.raises(FileNotFoundError):
        SED.load(base_path)

# Basic test for gather_3d (more comprehensive tests would mock logger or check values)
def test_sed_gather_3d_smoke(valid_sed_data):
    """Smoke test for gather_3d method to ensure it runs and returns tuple of arrays."""
    sed_obj = SED(**valid_sed_data)
    result = sed_obj.gather_3d(intensity_thresh_rel=0.01)
    assert isinstance(result, tuple)
    assert len(result) == 4
    for arr in result:
        assert isinstance(arr, np.ndarray)

def test_sed_gather_3d_empty_input():
    sed_obj = SED(np.array([]).reshape(0,0,3), np.array([]), np.array([]), np.array([]).reshape(0,3))
    kx, ky, freqs, amps = sed_obj.gather_3d()
    assert kx.size == 0 and ky.size == 0 and freqs.size == 0 and amps.size == 0 