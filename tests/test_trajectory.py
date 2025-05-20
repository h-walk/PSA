import pytest
import numpy as np
from sda.core.trajectory import Trajectory

@pytest.fixture
def valid_trajectory_data():
    """Provides a minimal set of valid data for Trajectory initialization."""
    n_frames = 2
    n_atoms = 3
    positions = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
    velocities = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
    types = np.ones(n_atoms, dtype=np.int32)
    timesteps = np.arange(n_frames, dtype=np.float32)
    box_matrix = np.eye(3, dtype=np.float32) * 10
    box_lengths = np.array([10, 10, 10], dtype=np.float32)
    box_tilts = np.zeros(3, dtype=np.float32)
    return {
        "positions": positions,
        "velocities": velocities,
        "types": types,
        "timesteps": timesteps,
        "box_matrix": box_matrix,
        "box_lengths": box_lengths,
        "box_tilts": box_tilts,
    }

def test_trajectory_initialization(valid_trajectory_data):
    """Test successful Trajectory initialization with valid data."""
    traj = Trajectory(**valid_trajectory_data)
    assert traj.n_frames == valid_trajectory_data["timesteps"].shape[0]
    assert traj.n_atoms == valid_trajectory_data["types"].shape[0]
    np.testing.assert_array_equal(traj.positions, valid_trajectory_data["positions"])
    np.testing.assert_array_equal(traj.box_matrix, valid_trajectory_data["box_matrix"])

# Test cases for initialization validation
@pytest.mark.parametrize("field_to_modify, invalid_value, error_message_part", [
    ("positions", np.random.rand(2, 3, 2), "Positions must be 3D"), # Incorrect pos shape
    ("velocities", np.random.rand(2, 3), "Velocities must be 3D"),    # Incorrect vel shape
    ("types", np.random.rand(2,3), "Types must be 1D"),
    ("timesteps", np.random.rand(2,3), "Timesteps must be 1D"),
    ("positions", np.random.rand(3, 3, 3), "Frame count mismatch"), # Mismatch n_frames
    ("types", np.ones(4), "Atom count mismatch"), # Mismatch n_atoms
    ("box_matrix", np.eye(2), "Box matrix must be 3x3"),
    ("box_lengths", np.array([10,10]), "Box lengths must be a 3-element array"),
    ("box_tilts", np.array([0,0]), "Box tilts must be a 3-element array"),
])
def test_trajectory_initialization_validation(valid_trajectory_data, field_to_modify, invalid_value, error_message_part):
    """Test Trajectory initialization raises ValueError for invalid data."""
    data = valid_trajectory_data.copy()
    # Special handling for frame/atom count mismatches
    if error_message_part == "Frame count mismatch":
        data["velocities"] = np.random.rand(data["positions"].shape[0] -1, data["types"].shape[0], 3)
    elif error_message_part == "Atom count mismatch":
         data["velocities"] = np.random.rand(data["timesteps"].shape[0], data["types"].shape[0] + 1, 3)
    
    data[field_to_modify] = invalid_value
    
    with pytest.raises(ValueError, match=error_message_part):
        Trajectory(**data)

def test_trajectory_properties(valid_trajectory_data):
    """Test n_frames and n_atoms properties."""
    traj = Trajectory(**valid_trajectory_data)
    assert traj.n_frames == 2
    assert traj.n_atoms == 3 