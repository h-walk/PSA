import pytest
import numpy as np
from sda.utils.helpers import parse_direction

# Test cases for string inputs
@pytest.mark.parametrize("input_str, expected_output", [
    ("x", np.array([1, 0, 0], dtype=np.float32)),
    ("y", np.array([0, 1, 0], dtype=np.float32)),
    ("z", np.array([0, 0, 1], dtype=np.float32)),
    ("100", np.array([1, 0, 0], dtype=np.float32)),
    ("xy", np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=np.float32)),
    ("110", np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=np.float32)),
    ("xyz", np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=np.float32)),
    ("111", np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=np.float32)),
    ("0,1,0", np.array([0, 1, 0], dtype=np.float32)),
    (" 1 0 0 ", np.array([1, 0, 0], dtype=np.float32)),
])
def test_parse_direction_string(input_str, expected_output):
    np.testing.assert_allclose(parse_direction(input_str), expected_output, atol=1e-6)

# Test cases for angle inputs (int/float)
@pytest.mark.parametrize("input_angle, expected_output", [
    (0, np.array([1, 0, 0], dtype=np.float32)),
    (90, np.array([0, 1, 0], dtype=np.float32)),
    (45, np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0], dtype=np.float32)),
    ("180.0", np.array([-1, 0, 0], dtype=np.float32)), # Angle as string
])
def test_parse_direction_angle(input_angle, expected_output):
    np.testing.assert_allclose(parse_direction(input_angle), expected_output, atol=1e-6)

# Test cases for list/tuple/array inputs
@pytest.mark.parametrize("input_vec, expected_output", [
    ([1,0,0], np.array([1,0,0], dtype=np.float32)),
    ((0,5,0), np.array([0,1,0], dtype=np.float32)), # Normalization
    (np.array([1,1,1]), np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=np.float32)),
    ([45], np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0], dtype=np.float32)), # Single element list (angle)
    (np.array(60.0), np.array([0.5, np.sqrt(3)/2, 0], dtype=np.float32)), # 0-dim array (angle)
])
def test_parse_direction_vector_like(input_vec, expected_output):
    np.testing.assert_allclose(parse_direction(input_vec), expected_output, atol=1e-6)

# Test cases for dictionary inputs
@pytest.mark.parametrize("input_dict, expected_output", [
    ({"angle": 30}, np.array([np.sqrt(3)/2, 0.5, 0], dtype=np.float32)),
    ({"h": 1, "k": 0, "l": 0}, np.array([1,0,0], dtype=np.float32)),
    ({"h": 1, "k": 1, "l": 0}, np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=np.float32)),
    ({"h": 0, "k": 0, "l": 2}, np.array([0,0,1], dtype=np.float32)), # Normalization
])
def test_parse_direction_dict(input_dict, expected_output):
    np.testing.assert_allclose(parse_direction(input_dict), expected_output, atol=1e-6)

# Test cases for invalid inputs
@pytest.mark.parametrize("invalid_input", [
    ("invalid_string"),
    ([1,2]), # Incorrect vector size
    ([1,2,3,4]), # Incorrect vector size
    np.array([[1,0,0],[0,1,0]]), # Incorrect dimensions
    ({"a":1, "b":2}), # Invalid dict keys
    (None),
    ([0,0,0]), # Zero vector
])
def test_parse_direction_invalid(invalid_input):
    if invalid_input is None:
        with pytest.raises(TypeError, match="Unsupported direction type: <class 'NoneType'>"):
            parse_direction(invalid_input)
    else:
        with pytest.raises(ValueError):
            parse_direction(invalid_input)

def test_parse_direction_zero_vector_behavior():
    """Test behavior for zero and near-zero vectors."""
    # Test that a vector that is allclose to zero raises ValueError
    near_zero_vec_allclose = np.array([1e-8, 1e-9, 1e-10], dtype=np.float32) # Default atol for allclose is 1e-8
    with pytest.raises(ValueError, match="Direction vector is zero."):
        parse_direction(near_zero_vec_allclose)

    # Test that an exact zero vector raises ValueError
    exact_zero_vec = np.array([0, 0, 0], dtype=np.float32)
    with pytest.raises(ValueError, match="Direction vector is zero."):
        parse_direction(exact_zero_vec)

    # Test a vector with very small norm but not allclose to [0,0,0] (if possible, depends on np.allclose details)
    # This case should ideally return the vector and log a warning.
    # For simplicity in this test, we ensure it doesn't raise ValueError like an allclose-to-zero vector.
    # If it does, it implies `np.allclose` threshold needs to be considered.
    small_norm_vec = np.array([1e-7, 0, 0], dtype=np.float32) # Slightly larger than atol of allclose
    # Depending on np.allclose's rtol and atol, this might or might not be allclose to [0,0,0]
    # If not allclose, it should be normalized or returned as is if norm is too small for division.
    # If parse_direction is robust, it will handle this without error, possibly returning the vector itself if norm is too small for safe division.
    try:
        result = parse_direction(small_norm_vec)
        # If norm < 1e-9 but not allclose to zero, it returns the vector itself.
        # If norm >= 1e-9, it returns normalized vector.
        if np.linalg.norm(small_norm_vec) < 1e-9:
            np.testing.assert_array_equal(result, small_norm_vec) 
        else:
            np.testing.assert_allclose(result, small_norm_vec / np.linalg.norm(small_norm_vec), atol=1e-6)
    except ValueError as e:
        # This path should not be taken if small_norm_vec is not allclose to [0,0,0]
        assert "Direction vector is zero" not in str(e), "Small norm vector incorrectly treated as zero vector" 