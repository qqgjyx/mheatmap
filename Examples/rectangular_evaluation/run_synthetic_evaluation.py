# ruff: noqa: E402, I001

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
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
    ReorderResult,
    compute_curves,
    contiguous_cluster_scores,
    hierarchical_olo_reorder,
    marginal_sort_reorder,
    mwb_auc,
    normalized_two_sum,
    one_walk_reorder,
    tw_reorder,
)
from _rectangular_tw_common import diagonal_band_mass
from mheatmap import mosaic_heatmap

OUTPUT_DIR = REPO_ROOT / "data" / "rectangular_evaluation"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = OUTPUT_DIR / "processed"
CASE_DIR = OUTPUT_DIR / "synthetic_cases"

METHOD_ORDER = ("Original", "Marginal", "HC+OLO", "One-walk", "TW")
SUPER_BLOCKS = 5
ROW_SUBGROUPS_PER_SUPER = 2
WIDTH_GRID = np.linspace(0.02, 0.50, 25)
NUM_SEEDS = 5


@dataclass(frozen=True)
class SizeSpec:
    key: str
    name: str
    n_rows: int
    n_cols: int
    row_sub_min: int
    col_sub_min: int


@dataclass(frozen=True)
class FamilySpec:
    key: str
    name: str
    description: str
    generator: str
    col_subgroups_per_super: int
    unique_keep: float
    unique_prob: float
    unique_lambda: float
    paired_keep: float = 0.0
    paired_prob: float = 0.0
    paired_lambda: float = 0.0
    shared_keep: float = 0.0
    shared_prob: float = 0.0
    shared_lambda: float = 0.0
    cross_keep: float = 0.0
    cross_prob: float = 0.0
    cross_lambda: float = 0.0
    off_target_choices: tuple[int, int] = (0, 0)
    off_target_lambda: float = 1.0


SIZES = (
    SizeSpec("small", "Small", n_rows=100, n_cols=90, row_sub_min=8, col_sub_min=5),
    SizeSpec(
        "medium",
        "Medium",
        n_rows=200,
        n_cols=180,
        row_sub_min=12,
        col_sub_min=8,
    ),
    SizeSpec(
        "large",
        "Large",
        n_rows=400,
        n_cols=360,
        row_sub_min=20,
        col_sub_min=14,
    ),
)

FAMILIES = (
    FamilySpec(
        key="clean_block",
        name="Family A: Clean one-to-one",
        description=(
            "Control regime with subgroup-specific columns and negligible leakage."
        ),
        generator="clean",
        col_subgroups_per_super=2,
        unique_keep=0.82,
        unique_prob=0.72,
        unique_lambda=5.0,
        off_target_choices=(0, 1),
        off_target_lambda=1.0,
    ),
    FamilySpec(
        key="paired_overlap",
        name="Family B: Paired subgroup overlap",
        description=(
            "Rows mix a primary subgroup prototype with the paired subgroup "
            "inside the same super-block."
        ),
        generator="paired",
        col_subgroups_per_super=2,
        unique_keep=0.78,
        unique_prob=0.60,
        unique_lambda=4.8,
        paired_keep=0.72,
        paired_prob=0.34,
        paired_lambda=3.0,
        off_target_choices=(0, 1),
        off_target_lambda=1.0,
    ),
    FamilySpec(
        key="shared_super",
        name="Family C: Shared super-prototype",
        description=(
            "Each row subgroup combines a unique prototype with a super-block "
            "prototype shared by both row subgroups."
        ),
        generator="shared",
        col_subgroups_per_super=3,
        unique_keep=0.74,
        unique_prob=0.54,
        unique_lambda=4.4,
        shared_keep=0.78,
        shared_prob=0.42,
        shared_lambda=4.0,
        off_target_choices=(0, 1),
        off_target_lambda=1.0,
    ),
    FamilySpec(
        key="shared_super_noisy",
        name="Family D: Shared prototype with noise",
        description=(
            "Shared super-block prototype plus row-specific off-target leakage."
        ),
        generator="shared_noisy",
        col_subgroups_per_super=3,
        unique_keep=0.72,
        unique_prob=0.40,
        unique_lambda=4.2,
        paired_keep=0.60,
        paired_prob=0.0,
        paired_lambda=2.0,
        shared_keep=0.78,
        shared_prob=0.48,
        shared_lambda=4.8,
        off_target_choices=(0, 1),
        off_target_lambda=1.0,
    ),
    FamilySpec(
        key="cross_block_leakage",
        name="Family E: Cross-block leakage",
        description=(
            "Shared super-block prototype with occasional weak connections to "
            "the next super-block."
        ),
        generator="cross",
        col_subgroups_per_super=3,
        unique_keep=0.72,
        unique_prob=0.48,
        unique_lambda=4.0,
        shared_keep=0.74,
        shared_prob=0.40,
        shared_lambda=4.0,
        cross_keep=0.68,
        cross_prob=0.12,
        cross_lambda=2.2,
        off_target_choices=(0, 1),
        off_target_lambda=1.1,
    ),
)


def ensure_directories() -> None:
    for directory in (OUTPUT_DIR, FIGURES_DIR, PROCESSED_DIR, CASE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def split_sizes(
    total: int,
    n_groups: int,
    minimum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base = np.full(n_groups, minimum, dtype=int)
    remaining = total - int(base.sum())
    if remaining < 0:
        msg = f"Invalid group-size configuration: total={total}, minimum={minimum}"
        raise ValueError(msg)
    if remaining == 0:
        return base
    allocation = rng.multinomial(remaining, np.full(n_groups, 1.0 / n_groups))
    return base + allocation


def subgroup_intervals(sizes: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], np.cumsum(sizes[:-1])])
    return [
        (int(start), int(start + size))
        for start, size in zip(starts, sizes, strict=True)
    ]


def choose_prototype(
    cols: np.ndarray,
    keep_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(cols) == 0:
        return np.array([], dtype=int)
    size = min(len(cols), max(2, int(np.ceil(keep_fraction * len(cols)))))
    return np.sort(rng.choice(cols, size=size, replace=False))


def add_weighted_hits(
    matrix: np.ndarray,
    row_index: int,
    cols: np.ndarray,
    hit_probability: float,
    poisson_lambda: float,
    rng: np.random.Generator,
) -> None:
    if len(cols) == 0 or hit_probability <= 0:
        return
    mask = rng.random(len(cols)) < hit_probability
    selected = cols[mask]
    if len(selected) == 0:
        return
    matrix[row_index, selected] += 1 + rng.poisson(poisson_lambda, len(selected))


def build_case(
    family: FamilySpec,
    size: SizeSpec,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    row_super_sizes = split_sizes(
        size.n_rows,
        SUPER_BLOCKS,
        ROW_SUBGROUPS_PER_SUPER * size.row_sub_min,
        rng,
    )
    col_super_sizes = split_sizes(
        size.n_cols,
        SUPER_BLOCKS,
        family.col_subgroups_per_super * size.col_sub_min,
        rng,
    )

    row_sub_sizes: list[int] = []
    row_super_labels: list[int] = []
    row_sub_labels: list[int] = []
    for super_id, super_size in enumerate(row_super_sizes):
        sizes = split_sizes(
            int(super_size),
            ROW_SUBGROUPS_PER_SUPER,
            size.row_sub_min,
            rng,
        )
        for local_sub, subgroup_size in enumerate(sizes):
            row_sub_sizes.append(int(subgroup_size))
            row_super_labels.extend([super_id] * int(subgroup_size))
            row_sub_labels.extend(
                [super_id * ROW_SUBGROUPS_PER_SUPER + local_sub] * int(subgroup_size)
            )

    col_sub_sizes: list[int] = []
    col_super_labels: list[int] = []
    col_sub_labels: list[int] = []
    for super_id, super_size in enumerate(col_super_sizes):
        sizes = split_sizes(
            int(super_size),
            family.col_subgroups_per_super,
            size.col_sub_min,
            rng,
        )
        for local_sub, subgroup_size in enumerate(sizes):
            col_sub_sizes.append(int(subgroup_size))
            col_super_labels.extend([super_id] * int(subgroup_size))
            col_sub_labels.extend(
                [super_id * family.col_subgroups_per_super + local_sub]
                * int(subgroup_size)
            )

    row_sub_ranges = subgroup_intervals(np.array(row_sub_sizes, dtype=int))
    col_sub_ranges = subgroup_intervals(np.array(col_sub_sizes, dtype=int))
    matrix = np.zeros((size.n_rows, size.n_cols), dtype=float)

    unique_prototypes: dict[tuple[int, int], np.ndarray] = {}
    shared_prototypes: dict[int, np.ndarray] = {}
    cross_prototypes: dict[int, np.ndarray] = {}
    for super_id in range(SUPER_BLOCKS):
        col_base = super_id * family.col_subgroups_per_super
        for local_sub in range(ROW_SUBGROUPS_PER_SUPER):
            unique_cols = np.arange(*col_sub_ranges[col_base + local_sub], dtype=int)
            unique_prototypes[(super_id, local_sub)] = choose_prototype(
                unique_cols,
                family.unique_keep,
                rng,
            )
        if family.col_subgroups_per_super >= 3:
            shared_cols = np.arange(*col_sub_ranges[col_base + 2], dtype=int)
            shared_prototypes[super_id] = choose_prototype(
                shared_cols,
                family.shared_keep,
                rng,
            )
        else:
            shared_prototypes[super_id] = np.array([], dtype=int)

    if family.cross_keep > 0:
        for super_id in range(SUPER_BLOCKS):
            next_super = (super_id + 1) % SUPER_BLOCKS
            cross_prototypes[super_id] = shared_prototypes[next_super]
    else:
        cross_prototypes = {
            super_id: np.array([], dtype=int) for super_id in range(SUPER_BLOCKS)
        }

    for global_row_sub, (row_start, row_stop) in enumerate(row_sub_ranges):
        super_id = global_row_sub // ROW_SUBGROUPS_PER_SUPER
        local_row_sub = global_row_sub % ROW_SUBGROUPS_PER_SUPER
        pair_local = 1 - local_row_sub

        primary_proto = unique_prototypes[(super_id, local_row_sub)]
        paired_proto = choose_prototype(
            unique_prototypes[(super_id, pair_local)],
            family.paired_keep,
            rng,
        )
        shared_proto = shared_prototypes[super_id]
        cross_proto = cross_prototypes[super_id]
        targeted = np.unique(
            np.concatenate(
                [
                    primary_proto,
                    paired_proto,
                    shared_proto,
                    cross_proto,
                ]
            )
        )
        all_cols = np.arange(size.n_cols, dtype=int)
        non_target_cols = np.setdiff1d(all_cols, targeted, assume_unique=False)

        for row_index in range(row_start, row_stop):
            add_weighted_hits(
                matrix,
                row_index,
                primary_proto,
                family.unique_prob,
                family.unique_lambda,
                rng,
            )
            add_weighted_hits(
                matrix,
                row_index,
                paired_proto,
                family.paired_prob,
                family.paired_lambda,
                rng,
            )
            add_weighted_hits(
                matrix,
                row_index,
                shared_proto,
                family.shared_prob,
                family.shared_lambda,
                rng,
            )
            add_weighted_hits(
                matrix,
                row_index,
                cross_proto,
                family.cross_prob,
                family.cross_lambda,
                rng,
            )
            noise_low, noise_high = family.off_target_choices
            if noise_high > 0 and len(non_target_cols) > 0:
                noise_count = int(rng.integers(noise_low, noise_high + 1))
                if noise_count > 0:
                    noise_targets = rng.choice(
                        non_target_cols,
                        size=min(noise_count, len(non_target_cols)),
                        replace=False,
                    )
                    matrix[row_index, noise_targets] += 1 + rng.poisson(
                        family.off_target_lambda,
                        len(noise_targets),
                    )

    row_perm = rng.permutation(size.n_rows)
    col_perm = rng.permutation(size.n_cols)
    observed = matrix[row_perm][:, col_perm]

    return {
        "ground_truth": matrix,
        "observed": observed,
        "row_perm": row_perm,
        "col_perm": col_perm,
        "row_super_labels": np.array(row_super_labels, dtype=int),
        "row_sub_labels": np.array(row_sub_labels, dtype=int),
        "col_super_labels": np.array(col_super_labels, dtype=int),
        "col_sub_labels": np.array(col_sub_labels, dtype=int),
    }


def save_case_payload(
    family: FamilySpec,
    size: SizeSpec,
    seed: int,
    payload: dict[str, np.ndarray],
) -> Path:
    stem = CASE_DIR / f"{family.key}_{size.key}_seed{seed:02d}"
    np.savez_compressed(stem.with_suffix(".npz"), **payload)
    metadata = {
        "family": family.key,
        "size": size.key,
        "seed": seed,
        "shape": list(payload["observed"].shape),
        "nnz_observed": int(np.count_nonzero(payload["observed"])),
        "nnz_ground_truth": int(np.count_nonzero(payload["ground_truth"])),
        "params": asdict(family),
    }
    stem.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return stem.with_suffix(".npz")


def save_case_catalog(rows: list[dict[str, object]]) -> None:
    pd.DataFrame.from_records(rows).to_csv(
        PROCESSED_DIR / "synthetic_case_catalog.csv",
        index=False,
    )


def plot_representative_heatmaps(
    representative_store: dict[str, dict[str, np.ndarray]],
) -> None:
    columns = ("Ground truth", "Observed", "Marginal", "HC+OLO", "One-walk", "TW")
    fig, axes = plt.subplots(
        len(FAMILIES),
        len(columns),
        figsize=(24, 14),
        constrained_layout=True,
    )
    for row_index, family in enumerate(FAMILIES):
        matrices = representative_store[family.key]
        positive = np.concatenate(
            [matrix[matrix > 0] for matrix in matrices.values() if np.any(matrix > 0)]
        )
        norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
        for col_index, column in enumerate(columns):
            ax = axes[row_index, col_index]
            matrix = np.ma.masked_where(matrices[column] <= 0, matrices[column])
            image = ax.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="YlGnBu",
                norm=norm,
            )
            if row_index == 0:
                ax.set_title(column, fontsize=13)
            if col_index == 0:
                ax.set_ylabel(family.name, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Synthetic rectangular benchmark (medium size, seed 0)", fontsize=16)
    fig.savefig(
        FIGURES_DIR / "synthetic_representative_heatmaps.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_representative_tw_mosaic(
    representative_store: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(20, 5), constrained_layout=True)
    for ax, family in zip(axes, FAMILIES, strict=True):
        tw_matrix = representative_store[family.key]["TW"]
        positive = tw_matrix[tw_matrix > 0]
        mosaic_heatmap(
            tw_matrix,
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            cmap="YlGnBu",
            norm=LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive))),
            cbar=True,
            cbar_kws={"label": "Synthetic edge weight"},
            rasterized=True,
        )
        ax.set_title(f"{family.name}\nTW + Mosaic", fontsize=12)
        ax.set_xlabel("Column bins")
        ax.set_ylabel("Row bins")
        ax.xaxis.set_ticks_position("top")
    fig.savefig(
        FIGURES_DIR / "synthetic_tw_mosaic.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_mwb_curves(
    curve_store: dict[tuple[str, str], dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(
        len(SIZES),
        len(FAMILIES),
        figsize=(20, 10),
        constrained_layout=True,
    )
    for row_index, size in enumerate(SIZES):
        for col_index, family in enumerate(FAMILIES):
            ax = axes[row_index, col_index]
            curves = curve_store[(family.key, size.key)]
            for method in METHOD_ORDER:
                ax.plot(WIDTH_GRID, curves[method], label=method, linewidth=1.8)
            if row_index == 0:
                ax.set_title(family.name, fontsize=11)
            if col_index == 0:
                ax.set_ylabel(f"{size.name}\nMass within band", fontsize=10)
            if row_index == len(SIZES) - 1:
                ax.set_xlabel("Band width fraction")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
    axes[0, -1].legend(loc="lower right", fontsize=8)
    fig.savefig(
        FIGURES_DIR / "synthetic_mwb_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_metric_tables(records: list[dict[str, object]]) -> None:
    df = pd.DataFrame.from_records(records)
    df.to_csv(PROCESSED_DIR / "synthetic_metrics.csv", index=False)

    grouped = (
        df.groupby(
            ["family", "family_key", "size", "size_key", "method"],
            as_index=False,
        )
        .agg(
            two_sum_mean=("two_sum", "mean"),
            two_sum_std=("two_sum", "std"),
            band_mass_05_mean=("band_mass_05", "mean"),
            band_mass_10_mean=("band_mass_10", "mean"),
            band_mass_20_mean=("band_mass_20", "mean"),
            mwb_auc_mean=("mwb_auc", "mean"),
            ari_sub_mean=("ari_sub_mean", "mean"),
            nmi_sub_mean=("nmi_sub_mean", "mean"),
            ari_super_mean=("ari_super_mean", "mean"),
            nmi_super_mean=("nmi_super_mean", "mean"),
            runtime_mean=("runtime_s", "mean"),
        )
    )
    grouped.to_csv(PROCESSED_DIR / "synthetic_summary.csv", index=False)

    family_grouped = (
        df.groupby(["family", "family_key", "method"], as_index=False)
        .agg(
            two_sum_mean=("two_sum", "mean"),
            band_mass_10_mean=("band_mass_10", "mean"),
            mwb_auc_mean=("mwb_auc", "mean"),
            ari_sub_mean=("ari_sub_mean", "mean"),
            nmi_sub_mean=("nmi_sub_mean", "mean"),
            ari_super_mean=("ari_super_mean", "mean"),
            nmi_super_mean=("nmi_super_mean", "mean"),
        )
    )
    family_grouped.to_csv(PROCESSED_DIR / "synthetic_family_summary.csv", index=False)

    lines = [
        "# Synthetic Rectangular Benchmark Summary",
        "",
        (
            "| Family | Size | Method | 2-SUM | Band@10% | MWB-AUC | "
            "ARI(sub) | NMI(sub) | ARI(super) | NMI(super) | Runtime (s) |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sorted_grouped = grouped.sort_values(["family", "size", "method"])
    for row in sorted_grouped.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.size} | {row.method} | {row.two_sum_mean:.6f} | "
            f"{row.band_mass_10_mean:.3f} | {row.mwb_auc_mean:.3f} | "
            f"{row.ari_sub_mean:.3f} | {row.nmi_sub_mean:.3f} | "
            f"{row.ari_super_mean:.3f} | {row.nmi_super_mean:.3f} | "
            f"{row.runtime_mean:.3f} |"
        )
    (PROCESSED_DIR / "synthetic_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_directories()

    reorderers = {
        "Marginal": marginal_sort_reorder,
        "HC+OLO": hierarchical_olo_reorder,
        "One-walk": one_walk_reorder,
        "TW": tw_reorder,
    }

    records: list[dict[str, object]] = []
    curve_accumulator: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {
        (family.key, size.key): {method: [] for method in METHOD_ORDER}
        for family in FAMILIES
        for size in SIZES
    }
    representative_store: dict[str, dict[str, np.ndarray]] = {}
    case_catalog: list[dict[str, object]] = []

    for family in FAMILIES:
        for size in SIZES:
            for seed in range(NUM_SEEDS):
                payload = build_case(family, size, seed)
                case_path = save_case_payload(family, size, seed, payload)
                case_catalog.append(
                    {
                        "family": family.key,
                        "size": size.key,
                        "seed": seed,
                        "case_path": str(case_path.relative_to(REPO_ROOT)),
                        "shape": (
                            f"{payload['observed'].shape[0]}x"
                            f"{payload['observed'].shape[1]}"
                        ),
                        "nnz_observed": int(np.count_nonzero(payload["observed"])),
                        "nnz_ground_truth": int(
                            np.count_nonzero(payload["ground_truth"])
                        ),
                    }
                )

                observed = payload["observed"]
                observed_row_sub = payload["row_sub_labels"][payload["row_perm"]]
                observed_col_sub = payload["col_sub_labels"][payload["col_perm"]]
                observed_row_super = payload["row_super_labels"][payload["row_perm"]]
                observed_col_super = payload["col_super_labels"][payload["col_perm"]]

                results: dict[str, ReorderResult] = {
                    "Original": ReorderResult(
                        observed.copy(),
                        np.arange(observed.shape[0], dtype=int),
                        np.arange(observed.shape[1], dtype=int),
                    )
                }
                runtimes = {"Original": 0.0}

                for method, reorder in reorderers.items():
                    start = time.perf_counter()
                    result = reorder(observed)
                    runtimes[method] = time.perf_counter() - start
                    results[method] = result

                curves = compute_curves(
                    {method: result.matrix for method, result in results.items()},
                    WIDTH_GRID,
                )
                for method in METHOD_ORDER:
                    curve_accumulator[(family.key, size.key)][method].append(
                        curves[method]
                    )
                    result = results[method]
                    matrix = result.matrix
                    sub_scores = contiguous_cluster_scores(
                        matrix,
                        observed_row_sub[result.row_order],
                        observed_col_sub[result.col_order],
                    )
                    super_scores = contiguous_cluster_scores(
                        matrix,
                        observed_row_super[result.row_order],
                        observed_col_super[result.col_order],
                    )
                    records.append(
                        {
                            "family": family.name,
                            "family_key": family.key,
                            "size": size.name,
                            "size_key": size.key,
                            "seed": seed,
                            "method": method,
                            "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                            "two_sum": normalized_two_sum(matrix),
                            "band_mass_05": diagonal_band_mass(matrix, 0.05),
                            "band_mass_10": diagonal_band_mass(matrix, 0.10),
                            "band_mass_20": diagonal_band_mass(matrix, 0.20),
                            "mwb_auc": mwb_auc(matrix, WIDTH_GRID),
                            "ari_sub_row": sub_scores["ari_row"],
                            "ari_sub_col": sub_scores["ari_col"],
                            "ari_sub_mean": sub_scores["ari_mean"],
                            "nmi_sub_row": sub_scores["nmi_row"],
                            "nmi_sub_col": sub_scores["nmi_col"],
                            "nmi_sub_mean": sub_scores["nmi_mean"],
                            "ari_super_row": super_scores["ari_row"],
                            "ari_super_col": super_scores["ari_col"],
                            "ari_super_mean": super_scores["ari_mean"],
                            "nmi_super_row": super_scores["nmi_row"],
                            "nmi_super_col": super_scores["nmi_col"],
                            "nmi_super_mean": super_scores["nmi_mean"],
                            "runtime_s": runtimes[method],
                        }
                    )

                if size.key == "medium" and seed == 0:
                    representative_store[family.key] = {
                        "Ground truth": payload["ground_truth"],
                        "Observed": observed,
                        "Marginal": results["Marginal"].matrix,
                        "HC+OLO": results["HC+OLO"].matrix,
                        "One-walk": results["One-walk"].matrix,
                        "TW": results["TW"].matrix,
                    }

    curve_store = {
        (family.key, size.key): {
            method: np.mean(curve_accumulator[(family.key, size.key)][method], axis=0)
            for method in METHOD_ORDER
        }
        for family in FAMILIES
        for size in SIZES
    }

    save_case_catalog(case_catalog)
    save_metric_tables(records)
    plot_representative_heatmaps(representative_store)
    plot_representative_tw_mosaic(representative_store)
    plot_mwb_curves(curve_store)


if __name__ == "__main__":
    main()
