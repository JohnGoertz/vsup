"""
Color transformation functions for VSUP.

This module provides functions for converting between different color spaces
and applying uncertainty-based transformations to colors using CIELAB color space.
"""

import numpy as np
from skimage import color

def usl_transform(colors: np.ndarray, uncertainty: np.ndarray, smin: float = 0.0, lmax: float = 1.0) -> np.ndarray:
    """
    Uncertainty mapped to Chroma (a*, b*) and Lightness (L*).
    
    Parameters
    ----------
    colors : array-like
        Input RGB colors
    uncertainty : array-like
        Uncertainty values
    smin : float, optional
        Minimum saturation/chroma (0 to 1)
    lmax : float, optional
        Maximum lightness (0 to 1). The maximum lightness will be 100 * lmax
    """
    lab_colors = color.rgb2lab(colors[..., :3])
    
    # Scale down chroma (a* and b*) based on uncertainty, but keep above smin
    chroma_scale = smin + (1 - smin) * (1 - uncertainty[..., np.newaxis])
    lab_colors[..., 1:] *= chroma_scale
    
    # Adjust lightness (L*) - move towards white (100 * lmax) as uncertainty increases
    max_lightness = 100 * lmax
    lab_colors[..., 0] = lab_colors[..., 0] * (1 - uncertainty) + max_lightness * uncertainty
    
    return color.lab2rgb(lab_colors)

def us_transform(colors: np.ndarray, uncertainty: np.ndarray, smin: float = 0.0) -> np.ndarray:
    """
    Uncertainty mapped to Chroma (a*, b*).
    
    Parameters
    ----------
    colors : array-like
        Input RGB colors
    uncertainty : array-like
        Uncertainty values
    smin : float, optional
        Minimum saturation/chroma (0 to 1)
    """
    lab_colors = color.rgb2lab(colors[..., :3])
    
    # Scale down chroma (a* and b*) based on uncertainty, but keep above smin
    chroma_scale = smin + (1 - smin) * (1 - uncertainty[..., np.newaxis])
    lab_colors[..., 1:] *= chroma_scale
    
    return color.lab2rgb(lab_colors)

def ul_transform(colors: np.ndarray, uncertainty: np.ndarray, lmax: float = 1.0) -> np.ndarray:
    """
    Uncertainty mapped to Lightness (L*).
    
    Parameters
    ----------
    colors : array-like
        Input RGB colors
    uncertainty : array-like
        Uncertainty values
    lmax : float, optional
        Maximum lightness (0 to 1). The maximum lightness will be 100 * lmax
    """
    lab_colors = color.rgb2lab(colors[..., :3])
    
    # Adjust lightness (L*) - move towards white (100 * lmax) as uncertainty increases
    max_lightness = 100 * lmax
    lab_colors[..., 0] = lab_colors[..., 0] * (1 - uncertainty) + max_lightness * uncertainty
    
    return color.lab2rgb(lab_colors)