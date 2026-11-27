from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "rectangular_evaluation" / "processed"

PRIMARY_SORT = ["mwb_auc_mean", "two_sum_mean", "ari_sub_mean", "nmi_sub_mean"]
METHOD_ORDER = ["Original", "Marginal", "HC+OLO", "One-walk", "TW(best $\\alpha$)"]
METHOD_SHORT = {
    "Original": "O",
    "Marginal": "M",
    "HC+OLO": "HC",
    "One-walk": "OW",
    "TW(best $\\alpha$)": "TW$^{*}$",
}


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(OUTPUT_DIR / "synthetic_summary.csv")
    alpha_best = pd.read_csv(OUTPUT_DIR / "synthetic_alpha_best.csv")
    return summary, alpha_best


def choose_best_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    baselines = summary[
        summary["method"].isin(["Original", "Marginal", "HC+OLO", "One-walk"])
    ].copy()
    baselines = baselines.sort_values(
        [
            "family",
            "size",
            "mwb_auc_mean",
            "two_sum_mean",
            "ari_sub_mean",
            "nmi_sub_mean",
        ],
        ascending=[True, True, False, True, False, False],
    )
    best = baselines.groupby(["family", "size"], as_index=False).first()
    best = best.rename(
        columns={
            "method": "baseline_method",
            "two_sum_mean": "baseline_two_sum",
            "band_mass_10_mean": "baseline_band10",
            "mwb_auc_mean": "baseline_auc",
            "ari_sub_mean": "baseline_ari",
            "nmi_sub_mean": "baseline_nmi",
            "runtime_mean": "baseline_runtime",
        }
    )
    return best[
        [
            "family",
            "size",
            "baseline_method",
            "baseline_two_sum",
            "baseline_band10",
            "baseline_auc",
            "baseline_ari",
            "baseline_nmi",
            "baseline_runtime",
        ]
    ]


def build_main_table(summary: pd.DataFrame, alpha_best: pd.DataFrame) -> pd.DataFrame:
    baseline = choose_best_baseline(summary)
    tw = alpha_best.rename(
        columns={
            "alpha": "tw_alpha",
            "two_sum_mean": "tw_two_sum",
            "band_mass_10_mean": "tw_band10",
            "mwb_auc_mean": "tw_auc",
            "ari_sub_mean": "tw_ari",
            "nmi_sub_mean": "tw_nmi",
            "runtime_mean": "tw_runtime",
        }
    )
    merged = baseline.merge(
        tw[
            [
                "family",
                "size",
                "tw_alpha",
                "tw_two_sum",
                "tw_band10",
                "tw_auc",
                "tw_ari",
                "tw_nmi",
                "tw_runtime",
            ]
        ],
        on=["family", "size"],
        how="inner",
    )
    merged["tw_beats_baseline_auc"] = merged["tw_auc"] > merged["baseline_auc"]
    merged["tw_beats_baseline_two_sum"] = (
        merged["tw_two_sum"] < merged["baseline_two_sum"]
    )
    merged["tw_beats_baseline_ari"] = merged["tw_ari"] > merged["baseline_ari"]
    merged["tw_beats_baseline_nmi"] = merged["tw_nmi"] > merged["baseline_nmi"]
    size_order = {"Small": 0, "Medium": 1, "Large": 2}
    merged["size_order"] = merged["size"].map(size_order)
    merged = merged.sort_values(["family", "size_order"]).drop(columns="size_order")
    return merged


def format_metric_pair(tw: float, base: float, higher_is_better: bool) -> str:
    tw_fmt = f"{tw:.3f}" if higher_is_better else f"{tw:.4f}"
    base_fmt = f"{base:.3f}" if higher_is_better else f"{base:.4f}"
    if (tw > base and higher_is_better) or (tw < base and not higher_is_better):
        tw_fmt = f"\\textbf{{{tw_fmt}}}"
    elif (tw < base and higher_is_better) or (tw > base and not higher_is_better):
        base_fmt = f"\\textbf{{{base_fmt}}}"
    return tw_fmt, base_fmt


def make_latex_table(main_table: pd.DataFrame) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{ll l rr rr rr rr r}",
        "\\toprule",
        "& & & \\multicolumn{2}{c}{2-SUM $\\downarrow$} & "
        "\\multicolumn{2}{c}{MWB-AUC $\\uparrow$} & "
        "\\multicolumn{2}{c}{ARI $\\uparrow$} & "
        "\\multicolumn{2}{c}{NMI $\\uparrow$} & "
        "Best \\\\",
        "Family & Size & Baseline & Base & TW$^{*}$ & Base & TW$^{*}$ & "
        "Base & TW$^{*}$ & Base & TW$^{*}$ & $\\alpha$ \\\\",
        "\\midrule",
    ]

    family_counts = main_table["family"].value_counts().to_dict()
    emitted = set()
    for row in main_table.itertuples(index=False):
        family_cell = ""
        if row.family not in emitted:
            family_label = row.family.replace("Family ", "F")
            family_cell = (
                f"\\multirow{{{family_counts[row.family]}}}{{*}}{{{family_label}}}"
            )
            emitted.add(row.family)
        base_two, tw_two = format_metric_pair(
            row.baseline_two_sum,
            row.tw_two_sum,
            False,
        )
        base_auc, tw_auc = format_metric_pair(row.baseline_auc, row.tw_auc, True)
        base_ari, tw_ari = format_metric_pair(row.baseline_ari, row.tw_ari, True)
        base_nmi, tw_nmi = format_metric_pair(row.baseline_nmi, row.tw_nmi, True)
        lines.append(
            f"{family_cell} & {row.size} & {row.baseline_method} & "
            f"{base_two} & {tw_two} & "
            f"{base_auc} & {tw_auc} & "
            f"{base_ari} & {tw_ari} & "
            f"{base_nmi} & {tw_nmi} & "
            f"{int(row.tw_alpha)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Synthetic rectangular benchmark. Each row reports the "
            "strongest non-TW baseline (chosen by MWB-AUC, with 2-SUM tie-break) "
            "against TW with the best $\\alpha \\in \\{2,4,6,8,12\\}$ for that "
            "family/size regime. ARI and NMI use contiguous subgroup recovery "
            "from the reordered matrix.}",
            "\\label{tab:synthetic-rectangular-main}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def make_full_csv(summary: pd.DataFrame, alpha_best: pd.DataFrame) -> pd.DataFrame:
    tw = alpha_best.copy()
    tw["method"] = "TW(best $\\alpha$)"
    tw = tw.rename(
        columns={
            "alpha": "best_alpha",
            "two_sum_mean": "two_sum_mean",
            "band_mass_10_mean": "band_mass_10_mean",
            "mwb_auc_mean": "mwb_auc_mean",
            "ari_sub_mean": "ari_sub_mean",
            "nmi_sub_mean": "nmi_sub_mean",
            "runtime_mean": "runtime_mean",
        }
    )
    tw = tw[
        [
            "family",
            "size",
            "method",
            "best_alpha",
            "two_sum_mean",
            "band_mass_10_mean",
            "mwb_auc_mean",
            "ari_sub_mean",
            "nmi_sub_mean",
            "runtime_mean",
        ]
    ]

    full = summary[
        [
            "family",
            "size",
            "method",
            "two_sum_mean",
            "band_mass_10_mean",
            "mwb_auc_mean",
            "ari_sub_mean",
            "nmi_sub_mean",
            "runtime_mean",
        ]
    ].copy()
    full["best_alpha"] = pd.NA
    full = pd.concat([full, tw], ignore_index=True, sort=False)
    size_order = {"Small": 0, "Medium": 1, "Large": 2}
    method_order = {
        "Original": 0,
        "Marginal": 1,
        "HC+OLO": 2,
        "One-walk": 3,
        "TW": 4,
        "TW(best $\\alpha$)": 5,
    }
    full["size_order"] = full["size"].map(size_order)
    full["method_order"] = full["method"].map(method_order).fillna(99)
    return full.sort_values(["family", "size_order", "method_order"]).drop(
        columns=["size_order", "method_order"]
    )


def abbreviate_family(name: str) -> str:
    return name.replace("Family ", "F")


def build_wide_table(summary: pd.DataFrame, alpha_best: pd.DataFrame) -> pd.DataFrame:
    full = make_full_csv(summary, alpha_best).copy()
    value_columns = [
        "two_sum_mean",
        "mwb_auc_mean",
        "runtime_mean",
        "ari_sub_mean",
        "nmi_sub_mean",
    ]
    wide = (
        full.pivot_table(
            index=["family", "size"],
            columns="method",
            values=value_columns,
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    wide.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_")
        for col in wide.columns.to_flat_index()
    ]
    size_order = {"Small": 0, "Medium": 1, "Large": 2}
    wide["size_order"] = wide["size"].map(size_order)
    return wide.sort_values(["family", "size_order"]).drop(columns="size_order")


def format_value(value: float, metric: str) -> str:
    if metric == "runtime_mean":
        return f"{value:.3f}"
    if metric == "two_sum_mean":
        return f"{value:.4f}"
    return f"{value:.3f}"


def maybe_bold(
    value: float,
    values: list[float],
    higher_is_better: bool,
    metric: str,
) -> str:
    target = max(values) if higher_is_better else min(values)
    formatted = format_value(value, metric)
    if abs(value - target) <= 1e-12:
        return f"\\textbf{{{formatted}}}"
    return formatted


def metric_block(
    row: pd.Series,
    metric: str,
    higher_is_better: bool,
) -> list[str]:
    values = [row[f"{metric}_{method}"] for method in METHOD_ORDER]
    return [
        maybe_bold(row[f"{metric}_{method}"], values, higher_is_better, metric)
        for method in METHOD_ORDER
    ]


def make_quality_latex_table(wide: pd.DataFrame, alpha_best: pd.DataFrame) -> str:
    alpha_lookup = {
        (row.family, row.size): int(row.alpha)
        for row in alpha_best.itertuples(index=False)
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll " + "r" * 16 + "}",
        "\\toprule",
        "& & \\multicolumn{5}{c}{2-SUM $\\downarrow$} & "
        "\\multicolumn{5}{c}{MWB-AUC $\\uparrow$} & "
        "\\multicolumn{5}{c}{Runtime (s) $\\downarrow$} & Best \\\\",
        "Family & Size & O & M & HC & OW & TW$^{*}$ & "
        "O & M & HC & OW & TW$^{*}$ & "
        "O & M & HC & OW & TW$^{*}$ & $\\alpha$ \\\\",
        "\\midrule",
    ]
    family_counts = wide["family"].value_counts().to_dict()
    emitted = set()
    for _, row in wide.iterrows():
        family = row["family"]
        size = row["size"]
        family_cell = ""
        if family not in emitted:
            family_cell = (
                f"\\multirow{{{family_counts[family]}}}{{*}}{{{abbreviate_family(family)}}}"
            )
            emitted.add(family)
        two_block = metric_block(row, "two_sum_mean", False)
        auc_block = metric_block(row, "mwb_auc_mean", True)
        run_block = metric_block(row, "runtime_mean", False)
        alpha = alpha_lookup[(family, size)]
        lines.append(
            f"{family_cell} & {size} & "
            + " & ".join(two_block + auc_block + run_block)
            + f" & {alpha} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Synthetic rectangular benchmark: ordering quality and "
            "runtime. Methods are Original (O), Marginal (M), HC+OLO (HC), "
            "One-walk (OW), and TW with the best "
            "$\\alpha \\in \\{2,4,6,8,12\\}$ for each regime. Bold numbers "
            "indicate the best method within each row/metric block.}",
            "\\label{tab:synthetic-rectangular-quality}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def make_recovery_latex_table(wide: pd.DataFrame, alpha_best: pd.DataFrame) -> str:
    alpha_lookup = {
        (row.family, row.size): int(row.alpha)
        for row in alpha_best.itertuples(index=False)
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll " + "r" * 11 + "}",
        "\\toprule",
        "& & \\multicolumn{5}{c}{ARI $\\uparrow$} & "
        "\\multicolumn{5}{c}{NMI $\\uparrow$} & Best \\\\",
        "Family & Size & O & M & HC & OW & TW$^{*}$ & "
        "O & M & HC & OW & TW$^{*}$ & $\\alpha$ \\\\",
        "\\midrule",
    ]
    family_counts = wide["family"].value_counts().to_dict()
    emitted = set()
    for _, row in wide.iterrows():
        family = row["family"]
        size = row["size"]
        family_cell = ""
        if family not in emitted:
            family_cell = (
                f"\\multirow{{{family_counts[family]}}}{{*}}{{{abbreviate_family(family)}}}"
            )
            emitted.add(family)
        ari_block = metric_block(row, "ari_sub_mean", True)
        nmi_block = metric_block(row, "nmi_sub_mean", True)
        alpha = alpha_lookup[(family, size)]
        lines.append(
            f"{family_cell} & {size} & "
            + " & ".join(ari_block + nmi_block)
            + f" & {alpha} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Synthetic rectangular benchmark: subgroup recovery "
            "metrics. ARI and NMI are computed from contiguous subgroup "
            "partitions induced by the reordered matrix. Methods are "
            "Original (O), Marginal (M), HC+OLO (HC), One-walk (OW), and TW "
            "with the best $\\alpha \\in \\{2,4,6,8,12\\}$. Bold numbers "
            "indicate the best method within each row/metric block.}",
            "\\label{tab:synthetic-rectangular-recovery}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    summary, alpha_best = load_tables()
    main_table = build_main_table(summary, alpha_best)
    main_table.to_csv(OUTPUT_DIR / "synthetic_results_main_table.csv", index=False)

    full_table = make_full_csv(summary, alpha_best)
    full_table.to_csv(OUTPUT_DIR / "synthetic_results_full_table.csv", index=False)

    wide = build_wide_table(summary, alpha_best)
    wide.to_csv(OUTPUT_DIR / "synthetic_results_wide_table.csv", index=False)

    latex = make_latex_table(main_table)
    (OUTPUT_DIR / "synthetic_results_main_table.tex").write_text(
        latex,
        encoding="utf-8",
    )
    quality_latex = make_quality_latex_table(wide, alpha_best)
    (OUTPUT_DIR / "synthetic_results_quality_table.tex").write_text(
        quality_latex,
        encoding="utf-8",
    )
    recovery_latex = make_recovery_latex_table(wide, alpha_best)
    (OUTPUT_DIR / "synthetic_results_recovery_table.tex").write_text(
        recovery_latex,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
