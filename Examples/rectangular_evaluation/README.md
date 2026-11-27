# Rectangular Evaluation

This folder contains two evaluation scripts for rectangular / bipartite matrices.

## Real Data

`run_realdata_evaluation.py` compares five methods on the three real-data examples:

- `Original`
- `Marginal`
- `HC+OLO`
- `One-walk`
- `TW`

Datasets:

- `1987 SIC -> 2002 NAICS`
- `ACS 2023 Massachusetts PUMS`
- `Tennessee LODES 2022`

Metrics:

- `2-SUM` (mass-normalized rectangular variant)
- `Band@5%`, `Band@10%`, `Band@20%`
- `MWB-AUC`
- `runtime`

Run:

```bash
python Examples/rectangular_evaluation/run_realdata_evaluation.py
```

## Synthetic

`run_synthetic_evaluation.py` builds a hierarchical rectangular synthetic benchmark with:

- `5` super-blocks
- `3` matrix sizes: `Small`, `Medium`, `Large`
- `5` synthetic families:
  - `Family A: Clean one-to-one`
  - `Family B: Paired subgroup overlap`
  - `Family C: Shared super-prototype`
  - `Family D: Shared prototype with noise`
  - `Family E: Cross-block leakage`

Each family/size regime is evaluated across multiple random seeds with the same five
methods as above. In addition to `2-SUM`, band-mass, and `MWB-AUC`, the script
records contiguous-segmentation recovery scores:

- `ARI(sub)` / `NMI(sub)`
- `ARI(super)` / `NMI(super)`

Run:

```bash
python Examples/rectangular_evaluation/run_synthetic_evaluation.py
```

## Synthetic Alpha Sweep

`run_synthetic_alpha_sweep.py` evaluates `TW(alpha)` against `One-walk` on the same
synthetic family/size grid, without modifying `src/`. The sweep uses:

- `alpha in {2, 4, 6, 8, 12}`

It writes per-seed metrics, aggregated summaries, and a best-alpha comparison for each
family/size regime.

Run:

```bash
python Examples/rectangular_evaluation/run_synthetic_alpha_sweep.py
```

Outputs are written under `data/rectangular_evaluation/`.
