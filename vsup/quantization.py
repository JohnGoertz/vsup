"""
Quantization functions for VSUP.

This module provides functions for quantizing value and uncertainty pairs
into discrete levels for visualization.
"""

import numpy as np


def linear_quantization(n_levels: int):
    """
    Create a linear quantization function that bins both value and uncertainty
    into n_levels discrete levels.

    Parameters
    ----------
    n_levels : int
        Number of quantization levels for both value and uncertainty

    Returns
    -------
    function
        A function that takes (value, uncertainty) arrays and returns
        quantized versions with values in [0, n_levels-1]
    """

    def quantize(
        value: np.ndarray, uncertainty: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Quantize value and uncertainty arrays into n_levels bins.

        Parameters
        ----------
        value : array-like
            Array of values to quantize
        uncertainty : array-like
            Array of uncertainty values to quantize

        Returns
        -------
        tuple
            (quantized_value, quantized_uncertainty) arrays with values in [0, n_levels-1]
        """
        # Ensure inputs are numpy arrays
        value = np.asarray(value)
        uncertainty = np.asarray(uncertainty)

        # Create bins for both value and uncertainty
        value_bins = np.linspace(0, 1, n_levels + 1)
        uncertainty_bins = np.linspace(0, 1, n_levels + 1)

        # Quantize value and uncertainty into bins
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
    into branching^layers discrete levels. The number of value bins is determined
    by the uncertainty level - higher uncertainty means fewer value bins.

    Parameters
    ----------
    branching : int
        Number of branches at each node
    layers : int
        Number of layers in the tree

    Returns
    -------
    function
        A function that takes (value, uncertainty) arrays and returns
        quantized versions with values in [0, branching^layers-1]
    """

    def quantize(
        value: np.ndarray, uncertainty: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Quantize value and uncertainty arrays using a tree structure.
        The number of value bins is determined by the uncertainty level.

        Parameters
        ----------
        value : array-like
            Array of values to quantize
        uncertainty : array-like
            Array of uncertainty values to quantize

        Returns
        -------
        tuple
            (quantized_value, quantized_uncertainty) arrays with values in [0, branching^layers-1]
        """
        # Ensure inputs are numpy arrays
        value = np.asarray(value)
        uncertainty = np.asarray(uncertainty)

        # Create bins for uncertainty
        uncertainty_bins = np.linspace(0, 1, layers + 1)

        # Quantize uncertainty into layers
        uncertainty_level = np.digitize(uncertainty, uncertainty_bins) - 1
        uncertainty_level = np.clip(uncertainty_level, 0, layers - 1)

        # For each uncertainty level, calculate number of value bins
        # Higher uncertainty means fewer value bins
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
                value_bins = np.linspace(0, 1, n_bins + 1)[:-1]

                # Quantize values for this uncertainty level
                quantized_value[mask] = (
                    np.digitize(value[mask], value_bins) - 0.5
                ) / n_bins
                quantized_uncertainty[mask] = level

        # Normalize to [0, 1] range for color mapping
        quantized_uncertainty = quantized_uncertainty / (layers - 1)

        return quantized_value, quantized_uncertainty

    return quantize
