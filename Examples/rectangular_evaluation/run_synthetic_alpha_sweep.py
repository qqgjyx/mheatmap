# ruff: noqa: E402, I001

from __future__ import annotations

import sys
import time
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

from _benchmark_utils import (
    contiguous_cluster_scores,
    mwb_auc,
    normalized_two_sum,
    one_walk_reorder,
    tw_alpha_reorder,
)
from _rectangular_tw_common import diagonal_band_mass
from run_synthetic_evaluation import FAMILIES, NUM_SEEDS, SIZES, WIDTH_GRID, build_case

OUTPUT_DIR = REPO_ROOT / "data" / "rectangular_evaluation"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = OUTPUT_DIR / "processed"
ALPHAS = (2.0, 4.0, 6.0, 8.0, 12.0)


def ensure_directories() -> None:
    for directory in (OUTPUT_DIR, FIGURES_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_tables(records: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    df.to_csv(PROCESSED_DIR / "synthetic_alpha_sweep.csv", index=False)

    summary = (
        df.groupby(
            ["family", "family_key", "size", "size_key", "method", "alpha"],
            dropna=False,
            as_index=False,
        )
        .agg(
            two_sum_mean=("two_sum", "mean"),
            band_mass_10_mean=("band_mass_10", "mean"),
            mwb_auc_mean=("mwb_auc", "mean"),
            ari_sub_mean=("ari_sub_mean", "mean"),
            nmi_sub_mean=("nmi_sub_mean", "mean"),
            ari_super_mean=("ari_super_mean", "mean"),
            nmi_super_mean=("nmi_super_mean", "mean"),
            runtime_mean=("runtime_s", "mean"),
        )
    )
    summary.to_csv(PROCESSED_DIR / "synthetic_alpha_summary.csv", index=False)

    tw_rows = summary[summary["method"] == "TW(alpha)"].copy()
    best_auc = (
        tw_rows.sort_values(
            ["family", "size", "mwb_auc_mean"],
            ascending=[True, True, False],
        )
        .groupby(["family", "size"], as_index=False)
        .first()
    )
    one_walk = (
        summary[summary["method"] == "One-walk"]
        .groupby(["family", "size"], as_index=False)
        .agg(
            one_walk_two_sum=("two_sum_mean", "first"),
            one_walk_band_10=("band_mass_10_mean", "first"),
            one_walk_mwb_auc=("mwb_auc_mean", "first"),
            one_walk_ari_sub=("ari_sub_mean", "first"),
            one_walk_nmi_sub=("nmi_sub_mean", "first"),
            one_walk_ari_super=("ari_super_mean", "first"),
            one_walk_nmi_super=("nmi_super_mean", "first"),
        )
    )
    best_auc = best_auc.merge(one_walk, on=["family", "size"], how="left")
    best_auc["beats_one_walk_auc"] = (
        best_auc["mwb_auc_mean"] > best_auc["one_walk_mwb_auc"]
    )
    best_auc["beats_one_walk_two_sum"] = (
        best_auc["two_sum_mean"] < best_auc["one_walk_two_sum"]
    )
    best_auc["beats_one_walk_ari_sub"] = (
        best_auc["ari_sub_mean"] > best_auc["one_walk_ari_sub"]
    )
    best_auc["beats_one_walk_nmi_sub"] = (
        best_auc["nmi_sub_mean"] > best_auc["one_walk_nmi_sub"]
    )
    best_auc.to_csv(PROCESSED_DIR / "synthetic_alpha_best.csv", index=False)

    family_best = (
        best_auc.groupby(["family"], as_index=False)
        .agg(
            best_alpha_mean=("alpha", "mean"),
            tw_auc_mean=("mwb_auc_mean", "mean"),
            ow_auc_mean=("one_walk_mwb_auc", "mean"),
            tw_two_sum_mean=("two_sum_mean", "mean"),
            ow_two_sum_mean=("one_walk_two_sum", "mean"),
            tw_ari_sub_mean=("ari_sub_mean", "mean"),
            ow_ari_sub_mean=("one_walk_ari_sub", "mean"),
            tw_nmi_sub_mean=("nmi_sub_mean", "mean"),
            ow_nmi_sub_mean=("one_walk_nmi_sub", "mean"),
        )
    )
    family_best.to_csv(PROCESSED_DIR / "synthetic_alpha_family_best.csv", index=False)

    lines = [
        "# Synthetic TW(alpha) Sweep Summary",
        "",
        (
            "| Family | Size | Best alpha | Best TW 2-SUM | Best TW Band@10% | "
            "Best TW MWB-AUC | Best TW ARI(sub) | Best TW NMI(sub) | "
            "One-walk 2-SUM | One-walk Band@10% | One-walk MWB-AUC | "
            "One-walk ARI(sub) | One-walk NMI(sub) | TW beats OW on AUC? | "
            "TW beats OW on 2-SUM? |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_auc.sort_values(["family", "size"]).itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.size} | {row.alpha:.0f} | {row.two_sum_mean:.6f} | "
            f"{row.band_mass_10_mean:.3f} | {row.mwb_auc_mean:.3f} | "
            f"{row.ari_sub_mean:.3f} | {row.nmi_sub_mean:.3f} | "
            f"{row.one_walk_two_sum:.6f} | {row.one_walk_band_10:.3f} | "
            f"{row.one_walk_mwb_auc:.3f} | {row.one_walk_ari_sub:.3f} | "
            f"{row.one_walk_nmi_sub:.3f} | {bool(row.beats_one_walk_auc)} | "
            f"{bool(row.beats_one_walk_two_sum)} |"
        )
    (PROCESSED_DIR / "synthetic_alpha_best.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def plot_alpha_curves(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(SIZES),
        len(FAMILIES),
        figsize=(20, 10),
        constrained_layout=True,
    )
    for row_index, size in enumerate(SIZES):
        for col_index, family in enumerate(FAMILIES):
            ax = axes[row_index, col_index]
            family_tw = summary[
                (summary["family"] == family.name)
                & (summary["size"] == size.name)
                & (summary["method"] == "TW(alpha)")
            ].sort_values("alpha")
            one_walk = summary[
                (summary["family"] == family.name)
                & (summary["size"] == size.name)
                & (summary["method"] == "One-walk")
            ].iloc[0]

            ax.plot(
                family_tw["alpha"],
                family_tw["mwb_auc_mean"],
                marker="o",
                linewidth=2,
                label="TW(alpha)",
            )
            ax.axhline(
                one_walk["mwb_auc_mean"],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="One-walk",
            )
            if row_index == 0:
                ax.set_title(family.name, fontsize=11)
            if col_index == 0:
                ax.set_ylabel(f"{size.name}\nMWB-AUC", fontsize=10)
            if row_index == len(SIZES) - 1:
                ax.set_xlabel("alpha")
            ax.set_xticks(ALPHAS)
            ax.grid(alpha=0.3)
    axes[0, -1].legend(loc="lower right", fontsize=8)
    fig.savefig(
        FIGURES_DIR / "synthetic_alpha_sweep_mwb_auc.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    ensure_directories()
    records: list[dict[str, object]] = []

    for family in FAMILIES:
        for size in SIZES:
            for seed in range(NUM_SEEDS):
                payload = build_case(family, size, seed)
                observed = payload["observed"]
                observed_row_sub = payload["row_sub_labels"][payload["row_perm"]]
                observed_col_sub = payload["col_sub_labels"][payload["col_perm"]]
                observed_row_super = payload["row_super_labels"][payload["row_perm"]]
                observed_col_super = payload["col_super_labels"][payload["col_perm"]]

                start = time.perf_counter()
                one_walk_result = one_walk_reorder(observed)
                one_walk_time = time.perf_counter() - start
                one_walk_sub = contiguous_cluster_scores(
                    one_walk_result.matrix,
                    observed_row_sub[one_walk_result.row_order],
                    observed_col_sub[one_walk_result.col_order],
                )
                one_walk_super = contiguous_cluster_scores(
                    one_walk_result.matrix,
                    observed_row_super[one_walk_result.row_order],
                    observed_col_super[one_walk_result.col_order],
                )
                records.append(
                    {
                        "family": family.name,
                        "family_key": family.key,
                        "size": size.name,
                        "size_key": size.key,
                        "seed": seed,
                        "method": "One-walk",
                        "alpha": np.nan,
                        "two_sum": normalized_two_sum(one_walk_result.matrix),
                        "band_mass_10": diagonal_band_mass(
                            one_walk_result.matrix,
                            0.10,
                        ),
                        "mwb_auc": mwb_auc(one_walk_result.matrix, WIDTH_GRID),
                        "ari_sub_mean": one_walk_sub["ari_mean"],
                        "nmi_sub_mean": one_walk_sub["nmi_mean"],
                        "ari_super_mean": one_walk_super["ari_mean"],
                        "nmi_super_mean": one_walk_super["nmi_mean"],
                        "runtime_s": one_walk_time,
                    }
                )

                for alpha in ALPHAS:
                    start = time.perf_counter()
                    tw_result = tw_alpha_reorder(observed, alpha=alpha)
                    runtime = time.perf_counter() - start
                    tw_sub = contiguous_cluster_scores(
                        tw_result.matrix,
                        observed_row_sub[tw_result.row_order],
                        observed_col_sub[tw_result.col_order],
                    )
                    tw_super = contiguous_cluster_scores(
                        tw_result.matrix,
                        observed_row_super[tw_result.row_order],
                        observed_col_super[tw_result.col_order],
                    )
                    records.append(
                        {
                            "family": family.name,
                            "family_key": family.key,
                            "size": size.name,
                            "size_key": size.key,
                            "seed": seed,
                            "method": "TW(alpha)",
                            "alpha": alpha,
                            "two_sum": normalized_two_sum(tw_result.matrix),
                            "band_mass_10": diagonal_band_mass(tw_result.matrix, 0.10),
                            "mwb_auc": mwb_auc(tw_result.matrix, WIDTH_GRID),
                            "ari_sub_mean": tw_sub["ari_mean"],
                            "nmi_sub_mean": tw_sub["nmi_mean"],
                            "ari_super_mean": tw_super["ari_mean"],
                            "nmi_super_mean": tw_super["nmi_mean"],
                            "runtime_s": runtime,
                        }
                    )

    summary = save_tables(records)
    plot_alpha_curves(summary)


if __name__ == "__main__":
    main()
