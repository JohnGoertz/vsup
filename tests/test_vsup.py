"""
Tests for the VSUP package.
"""

import numpy as np
import pytest
from vsup import VSUP
from vsup.quantization import (
    linear_quantization,
    tree_quantization,
)


def test_scale_initialization():
    """Test VSUP initialization with different modes."""
    # Test valid modes
    for mode in ["usl", "us", "ul"]:
        scale = VSUP(mode=mode)
        assert scale.mode == mode

    # Test invalid mode
    with pytest.raises(ValueError):
        VSUP(mode="invalid")


@pytest.mark.parametrize(
    "quantization, expected_value, expected_uncert",
    [
        (linear_quantization(5), 0.75, 0.25),
        # (square_quantization(5), 0.6, 0.2),
        (tree_quantization(2, 3), 0.625, 0),
    ],
)
def test_quantization_functions(quantization, expected_value, expected_uncert):
    """Test quantization functions."""
    # Test linear quantization
    value, uncert = quantization(0.7, 0.3)

    np.testing.assert_allclose(value, expected_value)
    np.testing.assert_allclose(uncert, expected_uncert)


def test_scale_color_mapping():
    """Test color mapping functionality."""
    scale = VSUP(mode="usl")

    # Test single value
    color = scale(0.5, 0.3)
    assert isinstance(color, np.ndarray)
    assert color.shape == (1, 3)  # RGBA color

    # Test array of values
    values = np.array([0.2, 0.5, 0.8])
    uncertainties = np.array([0.1, 0.3, 0.5])
    colors = scale(values, uncertainties)
    assert isinstance(colors, np.ndarray)
    assert colors.shape == (3, 3)  # 3 RGBA colors
