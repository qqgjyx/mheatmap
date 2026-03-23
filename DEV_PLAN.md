# mheatmap — Development Plan

> Living doc. Update as work progresses.

## Status: v1.2.5, Phase 1+2 complete

### What's Done

| Item | Status |
|------|--------|
| Repo scaffolding (src layout, pyproject.toml) | Done |
| PyPI name claimed (`mheatmap`) | Done |
| README + docs site (mkdocs-material) | Done |
| Mosaic heatmap visualization (`matrix.py`) | Done |
| AMC post-processing (`_amc_postprocess.py`) | Done |
| RMS permutation analysis (`_rms_permute.py`) | Done |
| Spectral reordering (`graph/_spectral_permute.py`) | Done |
| Two-walk Laplacian (`graph/_two_walk_laplacian.py`) | Done |
| Bipartite co-permutation (`graph/_copermute_from_bipermute.py`) | Done |
| Utility visualizations (plot_eigen, plot_bipartite) | Done |
| Test suite (4 files, 14 tests) | Done |
| API reference docs | Done |
| Example notebooks (mosaic, rms, spectral) | Done |
| Chinese documentation (`docs/.zh/`) | Done |
| `__init__.py` public API exports + `__all__` | Done |
| Migrate to uv (from pip/setuptools) | Done |
| Migrate to hatchling build backend | Done |
| Add Ruff linting + formatting | Done |
| Add pre-commit hooks | Done |
| Consolidate config (removed setup.py, setup.cfg, pytest.ini, .flake8, MANIFEST.in) | Done |
| Bump Python to ≥3.10, pin dependency versions | Done |
| CI: lint → test matrix (3 OS × 2 Python) → gated docs deploy | Done |
| CI: PyPI publish via uv on release | Done |
| Fix mkdocs.yml (removed duplicate arithmatex, MathJax, dead tags/mike config) | Done |
| Fix networkx dependency (lazy import, optional `[viz]` extra) | Done |

### What's Next

#### Phase 3: Test & Quality

1. **Expand test coverage** — target 80%+ line coverage
   - More edge cases for graph module (isolated vertices, 1×1 matrices)
   - AMC with various label configurations
   - RMS threshold boundary cases
2. **Add pytest-cov** reporting to CI pipeline
3. **Test mode improvements** — ensure all plotting functions respect test_mode

#### Phase 4: API & Features

1. **Promote utils to top-level** — decide whether `plot_eigen` and `plot_bipartite_confusion_matrix` should be in `__all__`
2. **Additional examples** — real-world confusion matrix datasets
3. **Performance profiling** — benchmark on large matrices (100×100+)

### Architecture Decisions

- **hatchling** build backend — simpler than setuptools, no setup.py/cfg/MANIFEST needed
- **Seaborn extension** — `_MosaicHeatMapper` extends seaborn's `_HeatMapper` for proportional cells
- **scipy.optimize.linear_sum_assignment** — Hungarian algorithm for label alignment in AMC
- **scipy.linalg.eigh** — eigendecomposition for spectral reordering (Fiedler vector)
- **Two-walk Laplacian** — captures bipartite structure better than standard Laplacian for asymmetric confusion matrices
- **Test mode decorator** — conditionally skips matplotlib rendering during tests
- **Lazy networkx import** — deferred to function call time to avoid breaking the package when networkx isn't installed

### Dev Environment

```bash
uv sync --extra dev     # install with dev deps
uv run pytest           # run tests
uv run ruff check src/  # lint
uv run mkdocs serve     # preview docs
```
