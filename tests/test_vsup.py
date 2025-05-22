"""
Tests for the VSUP package.
"""

import numpy as np
import pytest
from vsup import Scale, linear_quantization, square_quantization, tree_quantization

def test_scale_initialization():
    """Test Scale initialization with different modes."""
    # Test valid modes
    for mode in ['usl', 'us', 'ul']:
        scale = Scale(mode=mode)
        assert scale.mode == mode
    
    # Test invalid mode
    with pytest.raises(ValueError):
        Scale(mode='invalid')

def test_quantization_functions():
    """Test quantization functions."""
    # Test linear quantization
    lin_quant = linear_quantization(5)
    value, uncert = lin_quant(0.7, 0.3)
    assert value == 0.6  # Should be quantized to nearest 0.2
    assert uncert == 0.3  # Uncertainty should be unchanged
    
    # Test tree quantization
    tree_quant = tree_quantization(2, 3)
    value, uncert = tree_quant(0.7, 0.3)
    assert value == 0.75  # Should be quantized to nearest 1/8
    assert uncert == 0.33  # Should be quantized to nearest 1/3

def test_scale_color_mapping():
    """Test color mapping functionality."""
    scale = Scale(mode='usl')
    
    # Test single value
    color = scale(0.5, 0.3)
    assert isinstance(color, np.ndarray)
    assert color.shape == (4,)  # RGBA color
    
    # Test array of values
    values = np.array([0.2, 0.5, 0.8])
    uncertainties = np.array([0.1, 0.3, 0.5])
    colors = scale(values, uncertainties)
    assert isinstance(colors, np.ndarray)
    assert colors.shape == (3, 4)  # 3 RGBA colors 