# <img src="docs/assets/mheatmap.png" width="40px" align="center" alt="mheatmap logo"> mheatmap

[![PyPI version](https://badge.fury.io/py/mheatmap.svg)](https://badge.fury.io/py/mheatmap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for advanced heatmap visualization and matrix analysis,
featuring mosaic/proportional heatmaps, confusion matrix post-processing,
and spectral reordering capabilities.

---

## 📋 Table of Contents

1. [🚀 Features](#-features)
2. [📦 Installation](#-installation)
    - [Install from PyPI](#install-from-pypi)
    - [Install from Source](#install-from-source)
3. [📘 Documentation](#-documentation)
4. [🛠 Contributing](#-contributing)
5. [📝 License](#-license)

---

## 🚀 Features

- **Mosaic Heatmap**  
  Visualize matrix values with proportionally-sized cells.  
  ![Comparison between normal and mosaic heatmap](examples/images/basic_mosaic_heatmap.png)

- **Automatic Model Calibration (AMC)**  
  Align, Mask, and Confusion—an algorithm for post-processing confusion matrices.

- **Spectral Reordering**  
  Reorder matrices based on spectral analysis.
  ![Spectral reordering example](examples/images/spectral_reordering.png)

- **RMS (Reverse Merge/Split) Analysis**  
  Perform advanced permutation analysis to explore matrix structures.  
  ![RMS permutation example](examples/images/rms_permutation.png)

---

## 📦 Installation

### Install from PyPI

```bash
pip install mheatmap
```

### Install from source

```bash
git clone https://github.com/qqgjyx/mheatmap.git
cd mheatmap
pip install .
```

## 📘 Documentation

Comprehensive documentation is available in the `docs/` directory,
including guides on:

- [Mosaic Heatmap](docs/mosaic_heatmap.md)
- [AMC Post-processing](docs/amc_postprocess.md)
- [Spectral Reordering](docs/spectral_reordering.md)
- [RMS Permutation](docs/rms_permutation.md)

## 🛠 Contributing

We welcome contributions to improve mheatmap! Please follow these steps:

1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit your changes
4. Open a pull request

## 📝 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
