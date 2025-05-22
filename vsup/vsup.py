"""
Value-Suppressing Uncertainty Palettes (VSUP) for visualizing data with uncertainty.

This module provides a class for creating and using VSUPs to visualize data with uncertainty
by mapping both value and uncertainty to color properties. VSUPs are a visualization technique
that encodes both the value and its uncertainty in a single color, making it easier to
understand the reliability of the data while maintaining the ability to distinguish values.

The module supports three visualization modes:
- USL: Uncertainty mapped to both Saturation and Lightness
- US: Uncertainty mapped to Saturation only
- UL: Uncertainty mapped to Lightness only

Note that transformations take place in the CIELAB color space: L-modes increase L* (lightness),
while S-modes decrease "chroma": a* (green-red) and b* (blue-yellow), equally.

It also provides two quantization methods:
- Linear: Independent binning of values and uncertainties
- Tree: Value bins depend on uncertainty level (higher uncertainty = fewer value bins)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Union, Optional, Literal, Tuple
from matplotlib.patches import Wedge, Arc
from matplotlib.collections import PatchCollection
from .quantization import linear_quantization, tree_quantization
from .transforms import usl_transform, us_transform, ul_transform


class VSUP:
    """
    Value-Suppressing Uncertainty Palette (VSUP).

    A class that combines quantization, scaling, and colorizing to create
    uncertainty-aware color mappings. VSUPs encode both value and uncertainty
    information in a single color, making it easier to understand the reliability
    of the data while maintaining the ability to distinguish values.

    The class supports three visualization modes:
    - USL: Uncertainty affects both saturation and lightness
    - US: Uncertainty affects saturation only
    - UL: Uncertainty affects lightness only

    Parameters
    ----------
    palette : str | matplotlib.colors.Colormap, optional
        Name of the palette to use (default: 'viridis'). Can be any matplotlib
        or seaborn colormap name.
    mode : {'usl', 'us', 'ul'}, optional
        Visualization mode to use:
        - 'usl': Uncertainty mapped to Saturation and Lightness (default)
        - 'us': Uncertainty mapped to Saturation
        - 'ul': Uncertainty mapped to Lightness
    quantization : {'linear', 'tree', None}, optional
        Type of quantization to use (default: 'linear'):
        - 'linear': Independent binning of values and uncertainties
        - 'tree': Value bins depend on uncertainty level
        - None: No quantization (continuous/256-bit color mapping)
    n_levels : int, optional
        Number of quantization levels for both value and uncertainty.
        Must be >= 2. Default is 5.
    tree_base : int, optional
        Branching factor for tree quantization (default: 2). Only used when
        quantization='tree'.
    vmin : float, optional
        Minimum value for colormapping. If None, will be set to min of first input.
    vmax : float, optional
        Maximum value for colormapping. If None, will be set to max of first input.
    umin : float, optional
        Minimum uncertainty for colormapping. If None, will be set to min of first input.
    umax : float, optional
        Maximum uncertainty for colormapping. If None, will be set to max of first input.
    smin : float, optional
        Minimum saturation/chroma (0 to 1). Default is 0.2. Higher values ensure
        colors remain visible even at high uncertainty.
    lmax : float, optional
        Maximum lightness (0 to 1). The maximum lightness will be 100 * lmax.
        Default is 0.9. Controls how white the colors become at high uncertainty.
    """

    def __init__(
        self,
        palette: Union[str, mcolors.Colormap] = "viridis",
        mode: Literal["usl", "us", "ul"] = "usl",
        quantization: Literal[None, "linear", "tree"] = "linear",
        n_levels: Optional[int] = 5,
        tree_base: int = 2,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        umin: Optional[float] = None,
        umax: Optional[float] = None,
        smin: float = 0.2,
        lmax: float = 0.9,
    ):
        # Validate input parameters
        if mode not in ["usl", "us", "ul"]:
            raise ValueError("mode must be one of 'usl', 'us', or 'ul'")
        if quantization not in [None, "linear", "tree"]:
            raise ValueError("quantization must be one of None, 'linear', or 'tree'")
        if not 0 <= smin <= 1:
            raise ValueError("smin must be between 0 and 1")
        if not 0 <= lmax <= 1:
            raise ValueError("lmax must be between 0 and 1")
        if n_levels is not None:
            if not isinstance(n_levels, int):
                raise TypeError("n_levels must be an integer")
            if n_levels < 2:
                raise ValueError("n_levels must be >= 2 if not None")

        # Store configuration parameters
        self.quantization = quantization
        self.mode = mode
        self.n_levels = n_levels
        self.tree_base = tree_base

        # Create or use colormap
        if isinstance(palette, str):
            if n_levels is None:
                self.cmap = sns.color_palette(palette, as_cmap=True)
            else:
                if quantization == "linear":
                    self.cmap = sns.color_palette(palette, as_cmap=True)
                else:
                    self.cmap = sns.color_palette(palette, as_cmap=True)
        elif isinstance(palette, mcolors.Colormap):
            self.cmap = palette
        else:
            raise TypeError(
                "colormap must be a string or matplotlib.colors.Colormap object"
            )

        # Store range parameters for normalization
        self.vmin = vmin
        self.vmax = vmax
        self.umin = umin
        self.umax = umax
        self.smin = smin
        self.lmax = lmax

        # Set up quantization function based on configuration
        if n_levels is None or quantization is None:
            self.quantize = lambda v, u: (v, u)  # No quantization
        elif quantization == "linear":
            self.quantize = linear_quantization(n_levels)
        else:  # tree
            self.quantize = tree_quantization(tree_base, n_levels)

    def _normalize(
        self, value: np.ndarray, uncertainty: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize values and uncertainties to [0,1] range.

        This internal method handles the normalization of input values and uncertainties
        to the [0,1] range based on the configured min/max values. If min/max values
        are not set, they are determined from the input data.

        Parameters
        ----------
        value : np.ndarray
            Array of values to normalize
        uncertainty : np.ndarray
            Array of uncertainty values to normalize

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Normalized value and uncertainty arrays in [0,1] range
        """
        # Set ranges if not already set
        if self.vmin is None:
            self.vmin = np.min(value)
        if self.vmax is None:
            self.vmax = np.max(value)
        if self.umin is None:
            self.umin = np.min(uncertainty)
        if self.umax is None:
            self.umax = np.max(uncertainty)

        # Normalize to [0,1]
        norm_value = (value - self.vmin) / (self.vmax - self.vmin)
        norm_uncertainty = (uncertainty - self.umin) / (self.umax - self.umin)

        return norm_value, norm_uncertainty

    def __call__(self, value: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        """
        Map values and uncertainties to colors.

        This method is the main interface for converting value-uncertainty pairs
        to colors. It handles the complete pipeline:
        1. Normalization of inputs to [0,1] range
        2. Quantization of values and uncertainties
        3. Base color mapping using the configured colormap
        4. Uncertainty-based color modifications

        Parameters
        ----------
        value : array-like
            Array of values to map. Can be of any dimension.
        uncertainty : array-like
            Array of uncertainty values to map. Must have the same shape as value.

        Returns
        -------
        np.ndarray
            Array of RGB colors with shape (..., 3) where ... represents the input array shape.
            For example, if input is shape (10, 20), output will be (10, 20, 3).
            If input is shape (5, 6, 7), output will be (5, 6, 7, 3).

        Raises
        ------
        ValueError
            If value and uncertainty arrays have different shapes
        """
        # Convert inputs to numpy arrays
        value = np.atleast_1d(value)
        uncertainty = np.atleast_1d(uncertainty)

        # Validate input shapes
        if value.shape != uncertainty.shape:
            raise ValueError(
                f"value and uncertainty must have the same shape. Got value shape {value.shape} and uncertainty shape {uncertainty.shape}"
            )

        # Store original shape for reshaping later
        original_shape = value.shape

        # Normalize values to [0,1] range
        norm_value, norm_uncertainty = self._normalize(value, uncertainty)

        # Quantize the normalized inputs
        quantized_value, quantized_uncertainty = self.quantize(
            norm_value, norm_uncertainty
        )

        # Get base colors from the colormap
        base_colors = self.cmap(quantized_value)

        # Apply uncertainty-based modifications with min/max constraints
        if self.mode == "usl":
            colors = usl_transform(
                base_colors, quantized_uncertainty, smin=self.smin, lmax=self.lmax
            )
        elif self.mode == "us":
            colors = us_transform(base_colors, quantized_uncertainty, smin=self.smin)
        else:  # ul
            colors = ul_transform(base_colors, quantized_uncertainty, lmax=self.lmax)

        # Ensure output has correct shape (original shape + 3 for RGB)
        if colors.shape != original_shape + (3,):
            raise ValueError(
                f"Returned colors had shape {colors.shape}, expected {original_shape + (3,)}"
            )

        return colors

    def create_simple_legend(
        self,
        ax: Optional[plt.Axes] = None,
        n_samples: int = 5,
        size: float = 1.0,
        title: str = "Value",
        label_format: str = "{:.1f}",
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Create a simple legend showing value colors with uncertainty levels.

        This method creates a basic legend that demonstrates how colors change
        with both value and uncertainty. It shows:
        1. A row of color samples for different values (with no uncertainty)
        2. A column of color samples showing how uncertainty affects a middle value

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot the legend on. If None, a new figure is created.
        n_samples : int, optional
            Number of value samples to show (default: 5). These are evenly
            spaced across the value range.
        size : float, optional
            Size of the legend elements (default: 1.0). This is a multiplier
            for the default scatter point size.
        title : str, optional
            Title for the legend (default: "Value")
        label_format : str, optional
            Format string for value labels (default: "{:.1f}"). This controls
            how the value numbers are displayed.

        Returns
        -------
        tuple
            (main_ax, legend_ax) containing the main plot axes and legend axes.
            These are the same axes object since this is a simple legend.
        """
        raise NotImplementedError("Simple legend not implemented yet")
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        # Create value samples - evenly spaced across the range
        values = np.linspace(0, 1, n_samples)
        uncertainties = np.zeros_like(values)  # Start with no uncertainty

        # Create legend elements - row of value samples
        for i, (v, u) in enumerate(zip(values, uncertainties)):
            # Get color for this value-uncertainty pair
            color = self(v, u)
            # Plot color sample
            ax.scatter([i], [0], c=[color], s=100 * size)
            # Add value label below
            ax.text(i, -0.2, label_format.format(v), ha="center")

        # Add uncertainty samples - column showing uncertainty effect
        uncertainty_values = np.linspace(0, 1, 3)  # Show 3 uncertainty levels
        for i, u in enumerate(uncertainty_values):
            # Use middle value to show uncertainty effect
            color = self(0.5, u)
            # Plot color sample
            ax.scatter([-1], [i], c=[color], s=100 * size)
            # Add uncertainty label
            ax.text(-1.5, i, label_format.format(u), ha="right")

        # Set up legend appearance
        ax.set_title(title)
        ax.set_xlim(-2, n_samples)
        ax.set_ylim(-0.5, 2.5)
        ax.axis("off")

        return ax, ax

    def create_heatmap_legend(
        self,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Create a heatmap-style legend showing value-uncertainty combinations.

        This method creates a 2D heatmap legend that shows how colors change
        across all combinations of values and uncertainties. The x-axis represents
        values and the y-axis represents uncertainties.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot the legend on. If None, a new figure is created.

        Returns
        -------
        ax
            The axes containing the heatmap legend.

        Notes
        -----
        The number of value and uncertainty levels is determined by the
        quantization settings:
        - For linear quantization: n_levels for both value and uncertainty
        - For tree quantization: branching^(n_levels-1) for value, n_levels for uncertainty
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Create value and uncertainty samples based on quantization settings
        if self.quantization == "linear":
            v_levels = self.n_levels
        else:
            v_levels = self.tree_base ** (self.n_levels - 1)
        # Create value samples at bin centers
        values = np.linspace(self.vmin, self.vmax, v_levels + 1)
        # Create uncertainty samples
        u_levels = self.n_levels
        uncertainties = np.linspace(self.umin, self.umax, u_levels + 1)

        # Create meshgrid for all value-uncertainty combinations
        V, U = np.meshgrid(values[:-1], uncertainties[:-1])

        # Get colors for all combinations
        colors = self(V, U)

        # Create heatmap
        ax.pcolormesh(values, uncertainties, colors, shading="flat")

        # Add labels

        ax.set_xlabel("Value")
        ax.set_ylabel("Uncertainty")
        ax.set_box_aspect(1)

        return ax

    def create_arcmap_legend(
        self,
        ax: Optional[plt.Axes] = None,
        angular_width=90,
        lines=False,
        line_color="0.9",
        line_alpha=1,
        line_width=3,
        orient="down",
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Create an arcmap-style legend showing value-uncertainty combinations using wedges.

        This method creates a radial legend where:
        - Angular position represents value
        - Radial distance represents uncertainty
        - Each wedge shows the color for a specific value-uncertainty combination

        This layout is particularly intuitive for tree-based quantization as it
        naturally shows the hierarchical relationship between value and uncertainty.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot the legend on. If None, a new figure is created.
        angular_width : int, optional
            Width of the arc in degrees (default: 90)
        lines : bool, optional
            Whether to show grid lines (default: False)
        line_color : str, optional
            Color of the grid lines (default: "0.9")
        line_alpha : float, optional
            Alpha value for grid lines (default: 1)
        line_width : float, optional
            Width of grid lines (default: 3)
        orient : {'up', 'left', 'down', 'right'}, optional
            Orientation of the arc (default: 'down')

        Returns
        -------
        ax
            The axes containing the arcmap legend.
        """
        if ax is None:
            ax = plt.subplots(figsize=(4, 4))[1]

        # Convert orientation to rotation angle
        rotation = {
            "up": 90,
            "left": 180,
            "down": 270,
            "right": 360,
        }[orient]

        # Create radial and angular divisions
        # Number of angular divisions depends on quantization
        n_angular = self.tree_base ** (self.n_levels - 1)
        value_step = (self.vmax - self.vmin) / n_angular
        # Create value samples at bin centers
        dummy_values = np.linspace(
            self.vmax - value_step / 2, self.vmin + value_step / 2, n_angular
        )
        # Convert angular width to radians and calculate start/stop angles
        angular_width = np.deg2rad(angular_width)
        start_angle = np.deg2rad(rotation) - angular_width / 2
        stop_angle = np.deg2rad(rotation) + angular_width / 2
        angular_edges = np.linspace(start_angle, stop_angle, n_angular + 1)

        # Create radial divisions for uncertainty levels
        n_radial = self.n_levels
        uncertainty_step = (self.umax - self.umin) / n_radial
        # Create uncertainty samples at bin centers
        dummy_uncertainties = np.linspace(
            self.umax - uncertainty_step / 2, self.umin + uncertainty_step / 2, n_radial
        )
        radial_edges = np.linspace(0, 1, n_radial + 1)

        # Create wedges for each sector
        wedges = []
        colors = []

        # Create wedges for each value-uncertainty combination
        for i in range(n_radial):
            r_inner = radial_edges[i]
            r_outer = radial_edges[i + 1]
            uncertainty = dummy_uncertainties[i]

            for j in range(n_angular):
                theta1 = np.degrees(angular_edges[j])
                theta2 = np.degrees(angular_edges[j + 1])
                value = dummy_values[j]

                # Create wedge for this sector
                wedge = Wedge((0, 0), r_outer, theta1, theta2, width=r_outer - r_inner)
                wedges.append(wedge)

                # Get color for this value-uncertainty pair
                color = self(value, uncertainty)
                colors.append(color)

        # Create patch collection for all wedges
        collection = PatchCollection(wedges, facecolors=colors, edgecolors="none")
        ax.add_collection(collection)

        if lines:
            # Add radial grid lines (concentric arcs)
            for r in radial_edges[1:]:
                arc = Arc(
                    xy=(0, 0),
                    width=2 * r,
                    height=2 * r,
                    angle=0,
                    theta1=np.rad2deg(start_angle),
                    theta2=np.rad2deg(stop_angle),
                    edgecolor=line_color,
                    alpha=line_alpha,
                    lw=line_width,
                )
                ax.add_artist(arc)

            # Add angular grid lines (radial lines)
            for layer in range(n_radial):
                r0 = radial_edges[layer]
                r1 = radial_edges[layer + 1]
                n_div = self.tree_base**layer + 1
                for theta in np.linspace(start_angle, stop_angle, n_div):
                    x0 = np.cos(theta) * r0
                    x1 = np.cos(theta) * r1
                    y0 = np.sin(theta) * r0
                    y1 = np.sin(theta) * r1
                    ax.plot(
                        [x0, x1],
                        [y0, y1],
                        color=line_color,
                        alpha=line_alpha,
                        lw=line_width,
                    )

        # Calculate offset for label positioning
        offset = np.sin(angular_width / n_angular)

        # Add uncertainty labels
        uncertainty_labels = np.linspace(self.umax, self.umin, n_radial + 1)
        for i, r in enumerate(radial_edges):
            angle = angular_edges[0]
            ha = "left"
            # Position labels based on orientation
            match orient:
                case "up":
                    x = np.cos(angle) * r + offset / 2
                    y = np.sin(angle) * r - offset / 2
                    labelrot = np.rad2deg(angle) - rotation
                case "left":
                    x = np.cos(angle) * r + offset / 2
                    y = np.sin(angle) * r + offset / 2
                    labelrot = np.rad2deg(angle) - 90
                case "down":
                    x = np.cos(angle) * r - offset / 2
                    y = np.sin(angle) * r + offset / 2
                    labelrot = np.rad2deg(angle) - 180
                case "right":
                    x = np.cos(angle) * r - offset / 2
                    y = np.sin(angle) * r - offset / 2
                    ha = "right"
                    labelrot = np.rad2deg(angle) - rotation

            # Add uncertainty label
            ax.text(
                x,
                y,
                f"{uncertainty_labels[i]:.2f}",
                ha=ha,
                va="center",
                rotation=labelrot,
                rotation_mode="anchor",
            )

        # Add value labels
        value_labels = np.linspace(
            self.vmax, self.vmin, (n_angular // self.tree_base) + 1
        )
        for j, angle in enumerate(angular_edges[:: self.tree_base]):
            x = np.cos(angle) * (1 + offset)
            y = np.sin(angle) * (1 + offset)
            ha = "center"
            va = "center"
            # Position labels based on orientation
            match orient:
                case "up":
                    labelrot = np.rad2deg(angle) - rotation
                case "left":
                    ha = "right"
                    labelrot = np.rad2deg(angle) - rotation
                case "down":
                    labelrot = rotation - np.rad2deg(angle)
                case "right":
                    ha = "left"
                    labelrot = rotation - np.rad2deg(angle)
            # Add value label
            ax.text(
                x,
                y,
                f"{value_labels[j]:.2f}",
                ha=ha,
                va=va,
                rotation=labelrot,
                rotation_mode="anchor",
            )

        # Set equal aspect ratio and limits based on orientation
        ax.set_aspect("equal")
        r_extent = np.sin(start_angle)
        match orient:
            case "up":
                ax.set_xlim(-r_extent * (1 + offset * 2), r_extent * (1 + offset * 2))
                ax.set_ylim(-offset * 2, 1 + offset * 2)
            case "left":
                ax.set_xlim(-(1 + offset * 2), offset * 2)
                ax.set_ylim(-r_extent * (1 + offset * 2), r_extent * (1 + offset * 2))
            case "down":
                ax.set_xlim(-r_extent * (1 + offset * 2), r_extent * (1 + offset * 2))
                ax.set_ylim(-(1 + offset * 2), offset * 2)
            case "right":
                ax.set_xlim(-offset * 2, 1 + offset * 2)
                ax.set_ylim(-r_extent * (1 + offset * 2), r_extent * (1 + offset * 2))

        ax.axis("off")
        return ax
