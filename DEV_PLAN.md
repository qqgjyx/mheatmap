# mheatmap — Development Plan

> Living doc. Update as work progresses.

## Status: v1.2.5, Phase 1+2 complete, Phase 3 planned

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

#### Phase 3: Code Refactoring (v1.3.0)

Full source code review and rewrite for best practices, minimalism, and correctness.

##### 3a — Critical Fixes

| # | Issue | File | Action |
|---|-------|------|--------|
| 1 | **`_HeatMapper` subclass** — depends on seaborn's private internal API; a seaborn update could silently break the package | `matrix.py` | Replace inheritance with standalone data-prep code. Extract DataFrame/ndarray handling, tick label processing, colormap defaults into own utilities. Build on matplotlib public API (`pcolormesh`, `colorbar`) only. |
| 2 | **Side-effect plotting in computation** — `spectral_permute()` calls `plot_eigen()` and `plot_bipartite_confusion_matrix()` unconditionally | `graph/_spectral_permute.py` | Separate computation from visualization. Make plotting opt-in via `plot=False` parameter or remove entirely. |
| 3 | **Return type mismatch** — signature says `-> tuple[3 arrays]`, docstring says 3 returns, but only `L_tw` is returned | `graph/_two_walk_laplacian.py` | Fix type annotation and docstring to match actual return value. |
| 4 | **Wrong import source** — `from matplotlib.pylab import eigh` should be `from scipy.linalg import eigh` | `graph/_spectral_permute.py` | Change to `scipy.linalg.eigh`. |

##### 3b — Simplification

| # | Issue | File | Action |
|---|-------|------|--------|
| 5 | **Unnecessary wrapper class** — `_AMCPostprocess` is instantiated once and discarded; Java-style getters add boilerplate | `_amc_postprocess.py` | Flatten into plain functions. Remove getter methods. |
| 6 | **Unnecessary wrapper class** — `_RMSPermute` same pattern; 5-tuple return is hard to read | `_rms_permute.py` | Flatten into plain functions. Use NamedTuple for self-documenting return. |
| 7 | **Global mutable test mode** — `_TEST_MODE_CONTAINER = [False]` is a workaround when pytest fixtures handle this cleanly | `constants.py`, `utils/_base.py` | Replace with env var `MHEATMAP_NO_PLOTS` + pytest fixture in `conftest.py`. |
| 8 | **Negated linear_sum_assignment** — `linear_sum_assignment(-conf_mat)` when `maximize=True` parameter exists (since scipy 0.17) | `_amc_postprocess.py` | Use `linear_sum_assignment(conf_mat, maximize=True)`. |
| 9 | **Dead code** — unused `linewidths`/`linecolor` params accepted but never used; `_original_rcParams` copy never referenced | `matrix.py`, `__init__.py` | Remove unused params (or implement them). Remove `_original_rcParams`. |
| 10 | **Magic threshold** — `0.37` in `_make_rms_map` has no explanation | `_rms_permute.py` | Document the threshold derivation or expose as parameter. |

##### 3c — Polish

| # | Issue | File | Action |
|---|-------|------|--------|
| 11 | **Redundant branches** — `_put_zero_rows_cols_tail` has two identical if/elif branches | `graph/_spectral_permute.py` | Consolidate into single early-return. |
| 12 | **Duplicated zero-detection** — `_get_B_sub` and `_put_zero_rows_cols_tail` both detect zero rows/cols independently | `graph/_spectral_permute.py` | Extract shared `_find_zero_rows_cols(B, threshold)` helper. |
| 13 | **MATLAB-style doc notation** — docstrings use `nrBsub x 1` instead of NumPy shape conventions | `graph/_copermute_from_bipermute.py` | Convert to NumPy doc convention. |

##### 3d — Test Improvements

| # | Issue | Action |
|---|-------|--------|
| 14 | **No conftest.py** | Create `tests/conftest.py` with `matplotlib.use("Agg")` + `plt.close("all")` autouse fixture. |
| 15 | **No parametrize** | Convert repetitive test methods to `@pytest.mark.parametrize`. |
| 16 | **Missing edge cases** | Add tests: empty matrices, 1×1, isolated vertices, division-by-zero paths. Target 80%+ coverage. |
| 17 | **Consider pytest-mpl** | Add to dev deps for visual regression testing (lower priority). |

#### Phase 4: API & Features

1. **Promote utils to top-level** — decide whether `plot_eigen` and `plot_bipartite_confusion_matrix` should be in `__all__`
2. **Additional examples** — real-world confusion matrix datasets
3. **Performance profiling** — benchmark on large matrices (100×100+)

### Architecture Decisions

- **hatchling** build backend — simpler than setuptools, no setup.py/cfg/MANIFEST needed
- **Seaborn extension** — `_MosaicHeatMapper` extends seaborn's `_HeatMapper` for proportional cells (**to be replaced in Phase 3a**)
- **scipy.optimize.linear_sum_assignment** — Hungarian algorithm for label alignment in AMC
- **scipy.linalg.eigh** — eigendecomposition for spectral reordering (Fiedler vector)
- **Two-walk Laplacian** — captures bipartite structure better than standard Laplacian for asymmetric confusion matrices
- **Test mode decorator** — conditionally skips matplotlib rendering during tests (**to be replaced in Phase 3b**)
- **Lazy networkx import** — deferred to function call time to avoid breaking the package when networkx isn't installed

### Dev Environment

```bash
uv sync --extra dev     # install with dev deps
uv run pytest           # run tests
uv run ruff check src/  # lint
uv run mkdocs serve     # preview docs
```
