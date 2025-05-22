"""
Quantization functions for VSUP.

This module provides functions for quantizing value and uncertainty pairs
into discrete levels for visualization. Quantization is an important step in
VSUP that helps reduce the complexity of the visualization while maintaining
the essential patterns in the data.

The module provides two quantization strategies:
1. Linear quantization: Independently bins values and uncertainties into
   equal-width intervals. This is simpler but may not capture the relationship
   between value and uncertainty.
2. Tree quantization: Uses a hierarchical approach where the number of value
   bins depends on the uncertainty level. Higher uncertainty means fewer value
   bins, reflecting the reduced confidence in precise value distinctions.
"""

import numpy as np
from typing import Tuple, List, Optional


def linear_quantization(n_levels: int):
    """
    Create a linear quantization function that bins both value and uncertainty
    into n_levels discrete levels.

    This function creates a quantizer that independently divides both the value
    and uncertainty ranges into n_levels equal-width bins. This is the simplest
    form of quantization and treats value and uncertainty as independent variables.

    Parameters
    ----------
    n_levels : int
        Number of quantization levels for both value and uncertainty.
        Must be >= 2.

    Returns
    -------
    function
        A function that takes (value, uncertainty) arrays and returns
        quantized versions with values in [0, 1] range.

    Notes
    -----
    The returned function performs the following steps:
    1. Creates equal-width bins for both value and uncertainty
    2. Assigns each input to its corresponding bin
    3. Normalizes the bin indices to [0, 1] range
    """

    def quantize(
        value: np.ndarray, uncertainty: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Quantize value and uncertainty arrays into n_levels bins.

        Parameters
        ----------
        value : array-like
            Array of values to quantize. Should be in [0, 1] range.
        uncertainty : array-like
            Array of uncertainty values to quantize. Should be in [0, 1] range.

        Returns
        -------
        tuple
            (quantized_value, quantized_uncertainty) arrays with values in [0, 1] range.
            The quantization preserves the relative ordering of values while
            reducing the number of distinct levels.
        """
        # Ensure inputs are numpy arrays
        value = np.asarray(value)
        uncertainty = np.asarray(uncertainty)

        # Create bins for both value and uncertainty
        # The bins are equal-width intervals in [0, 1]
        value_bins = np.linspace(0, 1, n_levels + 1)
        uncertainty_bins = np.linspace(0, 1, n_levels + 1)

        # Quantize value and uncertainty into bins
        # digitize returns the index of the bin for each value
        quantized_value = np.digitize(value, value_bins) - 1
        quantized_uncertainty = np.digitize(uncertainty, uncertainty_bins) - 1

        # Ensure values are in [0, n_levels-1]
        quantized_value = np.clip(quantized_value, 0, n_levels - 1)
        quantized_uncertainty = np.clip(quantized_uncertainty, 0, n_levels - 1)

        # Normalize to [0, 1] range for color mapping
        quantized_value = quantized_value / (n_levels - 1)
        quantized_uncertainty = quantized_uncertainty / (n_levels - 1)

        return quantized_value, quantized_uncertainty

    return quantize


def tree_quantization(branching: int, layers: int):
    """
    Create a tree quantization function that bins value and uncertainty
    into branching^layers discrete levels.

    This function creates a quantizer that uses a hierarchical approach where
    the number of value bins depends on the uncertainty level. Higher uncertainty
    means fewer value bins, reflecting the reduced confidence in precise value
    distinctions. This approach better captures the relationship between value
    and uncertainty.

    Parameters
    ----------
    branching : int
        Number of branches at each node. This determines how many value bins
        are created for each uncertainty level.
    layers : int
        Number of layers in the tree. This determines the number of uncertainty
        levels and the maximum number of value bins (branching^(layers-1)).

    Returns
    -------
    function
        A function that takes (value, uncertainty) arrays and returns
        quantized versions with values in [0, 1] range.

    Notes
    -----
    The returned function performs the following steps:
    1. Divides uncertainty range into 'layers' levels
    2. For each uncertainty level, creates branching^(layers-1-level) value bins
    3. Assigns values to bins based on their uncertainty level
    4. Normalizes the results to [0, 1] range
    """

    def quantize(
        value: np.ndarray, uncertainty: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Quantize value and uncertainty arrays using a tree structure.

        Parameters
        ----------
        value : array-like
            Array of values to quantize. Should be in [0, 1] range.
        uncertainty : array-like
            Array of uncertainty values to quantize. Should be in [0, 1] range.

        Returns
        -------
        tuple
            (quantized_value, quantized_uncertainty) arrays with values in [0, 1] range.
            The quantization preserves the relationship between value and uncertainty,
            with higher uncertainty leading to coarser value quantization.
        """
        # Ensure inputs are numpy arrays
        value = np.asarray(value)
        uncertainty = np.asarray(uncertainty)

        # Create bins for uncertainty
        # Divide uncertainty range into 'layers' levels
        uncertainty_bins = np.linspace(0, 1, layers + 1)

        # Quantize uncertainty into layers
        uncertainty_level = np.digitize(uncertainty, uncertainty_bins) - 1
        uncertainty_level = np.clip(uncertainty_level, 0, layers - 1)

        # For each uncertainty level, calculate number of value bins
        # Higher uncertainty means fewer value bins
        # For level i, we have branching^(layers-1-i) bins
        value_bins_per_level = branching ** (layers - 1 - uncertainty_level)

        # Initialize output arrays
        quantized_value = np.zeros_like(value)
        quantized_uncertainty = np.zeros_like(uncertainty)

        # Process each uncertainty level separately
        for level in range(layers):
            # Get mask for current uncertainty level
            mask = uncertainty_level == level

            if np.any(mask):
                # Calculate number of bins for this uncertainty level
                n_bins = value_bins_per_level[mask][0]  # Same for all in this level

                # Create value bins for this uncertainty level
                # We use [:-1] to exclude the right edge of the last bin
                value_bins = np.linspace(0, 1, n_bins + 1)[:-1]

                # Quantize values for this uncertainty level
                # We subtract 0.5 to center the values in their bins
                quantized_value[mask] = (
                    np.digitize(value[mask], value_bins) - 0.5
                ) / n_bins
                quantized_uncertainty[mask] = level

        # Normalize uncertainty to [0, 1] range for color mapping
        quantized_uncertainty = quantized_uncertainty / (layers - 1)

        return quantized_value, quantized_uncertainty

    return quantize
