# VSUP: Value-Suppressing Uncertainty Palettes

A Python package for visualizing data with uncertainty using Value-Suppressing Uncertainty Palettes (VSUPs). Inspired by [https://github.com/uwdata/vsup](https://github.com/uwdata/vsup).

## Installation

```bash
pip install vsup
```

### Development

The project is developed with [uv](https://docs.astral.sh/uv/).

To check for a local python environment, run:

```bash
uv run python
```

Also install the [pre-commit](https://pre-commit.com/) hooks with:

```bash
uv tool install pre-commit
pre-commit install
```

## Usage
```python
from vsup import VSUP
import numpy as np
import matplotlib.pyplot as plt

# Create a grid of values and uncertainties for better visualization
n_points = 50
step = 1 / n_points
values = np.linspace(step / 2, 1 - step / 2, n_points)
uncertainties = np.linspace(step / 2, 1 - step / 2, n_points)

# Create a 2D grid
values, uncertainties = np.meshgrid(values, uncertainties)

# Colorize the data
axs = plt.subplots(3, 3, figsize=(9, 9))[1]

for row, quantization in zip(axs, [None, "linear", "tree"]):
    for ax, mode in zip(row, ["us", "ul", "usl"]):
        vsup = VSUP(palette="flare", mode=mode, quantization=quantization)

        colors = vsup(values, uncertainties)
        ax.pcolormesh(values, uncertainties, colors)
        # ax.set_title(f"{mode}")  #\n({description})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Uncertainty")
```

## Example

```python {marimo}
import marimo as mo

import micropip
await micropip.install(["vsup", "matplotlib", "numpy"])

from matplotlib import colormaps

seaborn_colors = ["flare", "rocket", "crest", "mako", "coolwarm", "vlag", "icefire", "ch:s=-.2,r=.6", "ch:start=.2,rot=-.3", "husl"]
matplotlib_colors = list(colormaps)
colors = seaborn_colors + matplotlib_colors

palette = mo.ui.dropdown(options=colors, value="flare", label="Palette:")
palette
```

```python {marimo}
def palette_not_selected():
    return mo.md("Please select a palette first.")

def showcase_palette(palette: str):
    try:
        from vsup import VSUP
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        return mo.md("Packages are still loading, please wait a moment.")

    # Create a grid of values and uncertainties for better visualization
    n_points = 50
    step = 1/n_points
    values = np.linspace(step/2, 1-step/2, n_points)
    uncertainties = np.linspace(step/2, 1-step/2, n_points)

    # Create a 2D grid
    values, uncertainties = np.meshgrid(values, uncertainties)

    # Colorize the data
    size = 9
    fig, axs = plt.subplots(3, 3, figsize=(size, size))

    for row, quantization in zip(axs, [None, 'linear','tree']):
        for ax, mode in zip(row, ["us", "ul", "usl"]):

            vsup = VSUP(palette=palette, mode=mode, quantization=quantization)

            colors = vsup(values, uncertainties)
            ax.pcolormesh(values, uncertainties, colors)
            # ax.set_title(f"{mode}")  #\n({description})")
            ax.set_xlabel("Value")
            ax.set_ylabel("Uncertainty")

    fig.suptitle(f"VSUP: {palette} palette")
    plt.tight_layout()
    return fig

func = palette_not_selected if not palette.value else lambda : showcase_palette(palette.value)
func()
```

## Features

- Three visualization modes:
  - USL: Uncertainty mapped to Saturation (chroma) and Lightness
  - US: Uncertainty mapped to Saturation
  - UL: Uncertainty mapped to Lightness
- Two quantization mods:
  - Linear: independent binning of values and uncertainties
  - Tree: value bins depend on uncertainty bin: lower uncertainty, higher value resolution
- Support for any matplotlib and seaborn colormaps

## Citation

If you use this package in your research, please cite the original VSUP paper:

```
@inproceedings{2018-uncertainty-palettes,
 title = {Value-Suppressing Uncertainty Palettes},
 author = {Michael Correll AND Dominik Moritz AND Jeffrey Heer},
 booktitle = {ACM Human Factors in Computing Systems (CHI)},
 year = {2018},
 url = {http://idl.cs.washington.edu/papers/uncertainty-palettes},
}
```

## License

MIT License
