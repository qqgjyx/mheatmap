# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mheatmap is a Python package for advanced heatmap visualization and confusion matrix analysis. It provides mosaic/proportional heatmaps, automatic model calibration (AMC) post-processing, RMS (Reverse Merge/Split) permutation analysis, and graph-based spectral reordering. Published on PyPI as `mheatmap`, currently at v1.2.5.

## Commands

```bash
# Environment (uses uv, not pip/conda)
uv sync                          # Install core deps
uv sync --extra dev              # Install with dev tools (pytest, ruff, pre-commit)
uv sync --extra docs             # Install with docs tools (mkdocs)

# Testing
uv run pytest                    # Run all tests
uv run pytest --cov              # Run tests with coverage
uv run pytest tests/test_foo.py  # Run a single test file
uv run pytest -k "test_name"     # Run a specific test by name

# Linting & formatting
uv run ruff check src/ tests/    # Lint
uv run ruff check --fix src/     # Lint with autofix
uv run ruff format src/ tests/   # Format

# Docs
uv run mkdocs serve              # Local docs preview
```

## Architecture

**src layout** — all source lives under `src/mheatmap/`.

### Module map

| Module | Purpose |
|--------|---------|
| `matrix.py` | Mosaic heatmap visualization (extends seaborn's `_HeatMapper`) |
| `_amc_postprocess.py` | Align-Mask-Confusion post-processing for confusion matrices |
| `_rms_permute.py` | Reverse Merge/Split permutation analysis |
| `graph/_spectral_permute.py` | Spectral reordering via Fiedler vector (two-walk or standard Laplacian) |
| `graph/_two_walk_laplacian.py` | Two-walk Laplacian construction for bipartite graphs |
| `graph/_copermute_from_bipermute.py` | Bipartite embedding permutation → row/column permutations |
| `utils/plot_eigen.py` | Eigenvalue spectrum and eigenvector visualization |
| `utils/plot_bipartite_confusion_matrix.py` | Bipartite graph visualization (requires `mheatmap[viz]` for networkx) |
| `utils/_base.py` | Test mode decorator (skips plotting in tests) |
| `constants.py` | Global test mode flag (`set_test_mode`, `get_test_mode`) |

### Key dependencies

- **numpy/scipy**: core math, sparse matrices, eigendecomposition, linear_sum_assignment
- **matplotlib/seaborn**: visualization (seaborn's `_HeatMapper` is the base for mosaic heatmaps)
- **scikit-learn**: confusion_matrix, label alignment utilities

All five (numpy, scipy, matplotlib, seaborn, scikit-learn) are core dependencies, always installed.

### Optional dependencies

- **networkx** (`mheatmap[viz]`): required only for `plot_bipartite_confusion_matrix`

## Conventions

- **Python ≥ 3.10**, version `1.2.5`
- **hatchling** build backend, single `pyproject.toml` (no setup.cfg)
- **uv** for env/package management, `uv.lock` committed for reproducibility
- **Ruff** for linting+formatting: line length 88, rules E/F/W/B/I/UP/RUF
- **pre-commit** hooks: trailing whitespace, EOF fixer, YAML/TOML checks, large file guard (500KB), ruff
- CI runs lint → test (matrix: ubuntu/macOS/windows × Python 3.10/3.12) → docs deploy
- Docs use **mkdocs-material** with **KaTeX** (not MathJax), **awesome-pages** for navigation
- `uv.lock` is committed for reproducibility
