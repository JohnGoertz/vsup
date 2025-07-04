"""
## Color transformation functions for VSUP.

This module provides functions for converting between different color spaces
and applying uncertainty-based transformations to colors using CIELAB color space.
The transformations modify the color properties (saturation/chroma and lightness)
based on the uncertainty value, making it easier to understand the reliability
of the data while maintaining the ability to distinguish values.

The module provides three transformation modes:

1. USL: Uncertainty affects both saturation and lightness
2. US: Uncertainty affects saturation only
3. UL: Uncertainty affects lightness only

"""

import numpy as np
from skimage import color


def usl_transform(
    colors: np.ndarray, uncertainty: np.ndarray, smin: float = 0.0, lmax: float = 1.0
) -> np.ndarray:
    """
    Transform colors by mapping uncertainty to both saturation and lightness.

    This function modifies the input colors based on uncertainty by:
    1. Reducing saturation (chroma) as uncertainty increases
    2. Increasing lightness as uncertainty increases

    The transformation is performed in CIELAB color space, which provides
    better perceptual uniformity than RGB or HSV.

    Parameters
    ----------
    colors : array-like
        Input RGB colors in range [0, 1]
    uncertainty : array-like
        Uncertainty values in range [0, 1]
    smin : float, optional
        Minimum saturation/chroma (0 to 1). Higher values ensure colors
        remain visible even at high uncertainty. Default is 0.0.
    lmax : float, optional
        Maximum lightness (0 to 1). The maximum lightness will be 100 \* lmax.
        Controls how white the colors become at high uncertainty.
        Default is 1.0.

    Returns
    -------
    np.ndarray
        Transformed RGB colors in range [0, 1]
    """
    lab_colors = color.rgb2lab(colors[..., :3])

    # Scale down chroma (a* and b*) based on uncertainty, but keep above smin
    chroma_scale = smin + (1 - smin) * (1 - uncertainty[..., np.newaxis])
    lab_colors[..., 1:] *= chroma_scale

    # Adjust lightness (L*) - move towards white (100 * lmax) as uncertainty increases
    max_lightness = 100 * lmax
    lab_colors[..., 0] = (
        lab_colors[..., 0] * (1 - uncertainty) + max_lightness * uncertainty
    )

    return color.lab2rgb(lab_colors)


def us_transform(
    colors: np.ndarray, uncertainty: np.ndarray, smin: float = 0.0
) -> np.ndarray:
    """
    Transform colors by mapping uncertainty to saturation only.

    This function modifies the input colors based on uncertainty by:
    1. Reducing saturation (chroma) as uncertainty increases
    2. Keeping lightness constant

    The transformation is performed in CIELAB color space, which provides
    better perceptual uniformity than RGB or HSV.

    Parameters
    ----------
    colors : array-like
        Input RGB colors in range [0, 1]
    uncertainty : array-like
        Uncertainty values in range [0, 1]
    smin : float, optional
        Minimum saturation/chroma (0 to 1). Higher values ensure colors
        remain visible even at high uncertainty. Default is 0.0.

    Returns
    -------
    np.ndarray
        Transformed RGB colors in range [0, 1]
    """
    lab_colors = color.rgb2lab(colors[..., :3])

    # Scale down chroma (a* and b*) based on uncertainty, but keep above smin
    chroma_scale = smin + (1 - smin) * (1 - uncertainty[..., np.newaxis])
    lab_colors[..., 1:] *= chroma_scale

    return color.lab2rgb(lab_colors)


def ul_transform(
    colors: np.ndarray, uncertainty: np.ndarray, lmax: float = 1.0
) -> np.ndarray:
    """
    Transform colors by mapping uncertainty to lightness only.

    This function modifies the input colors based on uncertainty by:
    1. Keeping saturation (chroma) constant
    2. Increasing lightness as uncertainty increases

    The transformation is performed in CIELAB color space, which provides
    better perceptual uniformity than RGB or HSV.

    Parameters
    ----------
    colors : array-like
        Input RGB colors in range [0, 1]
    uncertainty : array-like
        Uncertainty values in range [0, 1]
    lmax : float, optional
        Maximum lightness (0 to 1). The maximum lightness will be 100 * lmax.
        Controls how white the colors become at high uncertainty.
        Default is 1.0.

    Returns
    -------
    np.ndarray
        Transformed RGB colors in range [0, 1]
    """
    lab_colors = color.rgb2lab(colors[..., :3])

    # Adjust lightness (L*) - move towards white (100 * lmax) as uncertainty increases
    max_lightness = 100 * lmax
    lab_colors[..., 0] = (
        lab_colors[..., 0] * (1 - uncertainty) + max_lightness * uncertainty
    )

    return color.lab2rgb(lab_colors)
