# VSUP: Value-Suppressing Uncertainty Palettes

A Python package for visualizing data with uncertainty using Value-Suppressing Uncertainty Palettes (VSUPs).

## Installation

```bash
pip install vsup
```

## Usage

```python
import vsup
import numpy as np

# Create a VSUP instance
vsup = VSUP(colormap='viridis', mode='usl')

# Generate some data with uncertainty
values = np.random.rand(100)
uncertainties = np.random.rand(100)

# Colorize the data
colors = vsup(values, uncertainties)
```

## Features

- Three visualization modes:
  - USL: Uncertainty mapped to Saturation and Lightness
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