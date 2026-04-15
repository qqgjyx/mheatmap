# ruff: noqa: E402, I001

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.modules.setdefault("numexpr", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "Examples"
EVAL_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, EXAMPLES_DIR, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from _benchmark_utils import (
    compute_curves,
    hierarchical_olo_reorder,
    load_matrix_csv,
    marginal_sort_reorder,
    mwb_auc,
    normalized_two_sum,
    one_walk_reorder,
    tw_reorder,
)
from _rectangular_tw_common import diagonal_band_mass

OUTPUT_DIR = REPO_ROOT / "data" / "rectangular_evaluation"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = OUTPUT_DIR / "processed"

METHOD_ORDER = (
    "Original",
    "Marginal",
    "HC+OLO",
    "One-walk",
    "TW",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    original_csv: Path


DATASETS = [
    DatasetSpec(
        key="naics_sic",
        name="1987 SIC -> 2002 NAICS",
        original_csv=REPO_ROOT
        / "data"
        / "naics_sic_rectangular"
        / "processed"
        / "sic1987_to_naics2002_original_matrix.csv",
    ),
    DatasetSpec(
        key="acs_pums",
        name="ACS 2023 MA PUMS OCCP x INDP",
        original_csv=REPO_ROOT
        / "data"
        / "acs_pums_rectangular"
        / "processed"
        / "acs2023_ma_occp_indp_original_matrix.csv",
    ),
    DatasetSpec(
        key="lodes",
        name="TN LODES 2022 Home x Work County",
        original_csv=REPO_ROOT
        / "data"
        / "lodes_rectangular"
        / "processed"
        / "tn_lodes2022_home_county_to_work_county_original_matrix.csv",
    ),
]


def ensure_directories() -> None:
    for directory in (OUTPUT_DIR, FIGURES_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_metrics_table(records: list[dict[str, object]]) -> None:
    df = pd.DataFrame.from_records(records)
    df.to_csv(PROCESSED_DIR / "realdata_metrics.csv", index=False)

    lines = [
        "# Real-Data Rectangular Reordering Evaluation",
        "",
        (
            "| Dataset | Method | Shape | 2-SUM | Band@5% | Band@10% | "
            "Band@20% | MWB-AUC | Runtime (s) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.method} | {row.shape} | "
            f"{row.two_sum:.6f} | {row.band_mass_05:.3f} | {row.band_mass_10:.3f} | "
            f"{row.band_mass_20:.3f} | {row.mwb_auc:.3f} | {row.runtime_s:.3f} |"
        )
    (PROCESSED_DIR / "realdata_metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot_mwb_curves(
    width_grid: np.ndarray,
    curve_store: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, spec in zip(axes, DATASETS, strict=True):
        curves = curve_store[spec.key]
        for method in METHOD_ORDER:
            ax.plot(width_grid, curves[method], label=method, linewidth=2)
        ax.set_title(spec.name, fontsize=11)
        ax.set_xlabel("Band width fraction")
        ax.set_ylabel("Mass within band")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.savefig(FIGURES_DIR / "realdata_mwb_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_grid(
    method_store: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(
        3,
        len(METHOD_ORDER),
        figsize=(24, 12),
        constrained_layout=True,
    )
    for row_index, spec in enumerate(DATASETS):
        matrices = method_store[spec.key]
        positive = np.concatenate(
            [matrix[matrix > 0] for matrix in matrices.values() if np.any(matrix > 0)]
        )
        norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
        for col_index, method in enumerate(METHOD_ORDER):
            ax = axes[row_index, col_index]
            matrix = np.ma.masked_where(matrices[method] <= 0, matrices[method])
            image = ax.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="YlGnBu",
                norm=norm,
            )
            if row_index == 0:
                ax.set_title(method, fontsize=14)
            if col_index == 0:
                ax.set_ylabel(spec.name, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(
        FIGURES_DIR / "realdata_method_heatmaps.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    ensure_directories()
    width_grid = np.linspace(0.02, 0.50, 25)

    records: list[dict[str, object]] = []
    curve_store: dict[str, dict[str, np.ndarray]] = {}
    method_store: dict[str, dict[str, np.ndarray]] = {}

    reorderers = {
        "Marginal": marginal_sort_reorder,
        "HC+OLO": hierarchical_olo_reorder,
        "One-walk": one_walk_reorder,
        "TW": tw_reorder,
    }

    for spec in DATASETS:
        original_matrix, _, _ = load_matrix_csv(spec.original_csv)

        matrices = {"Original": original_matrix}
        runtimes = {"Original": 0.0}

        for method, reorder in reorderers.items():
            start = time.perf_counter()
            result = reorder(original_matrix)
            runtimes[method] = time.perf_counter() - start
            matrices[method] = result.matrix

        curve_store[spec.key] = compute_curves(matrices, width_grid)
        method_store[spec.key] = matrices

        for method in METHOD_ORDER:
            matrix = matrices[method]
            records.append(
                {
                    "dataset": spec.name,
                    "method": method,
                    "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                    "two_sum": normalized_two_sum(matrix),
                    "band_mass_05": diagonal_band_mass(matrix, 0.05),
                    "band_mass_10": diagonal_band_mass(matrix, 0.10),
                    "band_mass_20": diagonal_band_mass(matrix, 0.20),
                    "mwb_auc": mwb_auc(matrix, width_grid),
                    "runtime_s": runtimes[method],
                }
            )

    save_metrics_table(records)
    plot_mwb_curves(width_grid, curve_store)
    plot_heatmap_grid(method_store)


if __name__ == "__main__":
    main()
