# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.modules.setdefault("numexpr", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "Examples"
for path in (SRC_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mheatmap.graph import spectral_permute

from _rectangular_tw_common import (
    diagonal_band_mass,
    plot_triptych,
    recover_column_labels,
    write_matrix_csv_fast,
    write_metadata_csv,
)

RAW_DIR = REPO_ROOT / "data" / "acs_pums_rectangular" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "acs_pums_rectangular" / "processed"
FIGURES_DIR = REPO_ROOT / "data" / "acs_pums_rectangular" / "figures"

SOURCE_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pma.zip"
RAW_FILENAME = "csv_pma.zip"
CSV_MEMBER = "psam_p25.csv"
OUTPUT_STEM = "acs2023_ma_occp_indp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ACS 2023 Massachusetts PUMS, build an occupation-by-industry "
            "matrix from weighted person records, select a low-entropy rectangular "
            "submatrix, and compare original / TW / TW+mosaic views."
        )
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def download_source(refresh: bool) -> Path:
    path = RAW_DIR / RAW_FILENAME
    if refresh or not path.exists():
        path.unlink(missing_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)
    return path


def load_person_data(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open(CSV_MEMBER) as handle:
            df = pd.read_csv(handle, usecols=["ESR", "OCCP", "INDP", "PWGTP"])

    df = df[df["ESR"].isin([1, 2])].dropna(subset=["OCCP", "INDP", "PWGTP"]).copy()
    df["OCCP"] = df["OCCP"].astype(int).astype(str).str.zfill(4)
    df["INDP"] = df["INDP"].astype(int).astype(str).str.zfill(4)
    return df


def find_best_submatrix(
    matrix_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    matrix = matrix_df.to_numpy(dtype=float)
    row_labels = matrix_df.index.to_numpy()
    col_labels = matrix_df.columns.to_numpy()

    row_totals = matrix.sum(axis=1)
    sorted_values = np.sort(matrix, axis=1)
    top2_share = (sorted_values[:, -1] + sorted_values[:, -2]) / row_totals
    best = None

    for min_total in (200, 500, 1000, 2000, 4000):
        for min_top2_share in (0.70, 0.75, 0.80, 0.85, 0.90):
            keep_rows = (row_totals >= min_total) & (top2_share >= min_top2_share)
            if keep_rows.sum() < 12:
                continue

            reduced = matrix[keep_rows]
            reduced_row_labels = row_labels[keep_rows]
            dominant_columns = np.argsort(reduced.sum(axis=0))[::-1]

            for n_cols in (10, 12, 15, 18, 20):
                if n_cols >= reduced.shape[1]:
                    continue

                col_indices = dominant_columns[:n_cols]
                reduced_cols = reduced[:, col_indices]
                active_rows = reduced_cols.sum(axis=1) > 0
                reduced_cols = reduced_cols[active_rows]
                reduced_row_labels_2 = reduced_row_labels[active_rows]
                if reduced_cols.shape[0] < n_cols + 4 or reduced_cols.shape[0] > 40:
                    continue

                row_mass = reduced_cols.sum(axis=1)
                row_sorted = np.sort(reduced_cols, axis=1)
                row_share = (row_sorted[:, -1] + row_sorted[:, -2]) / row_mass
                candidate_order = np.argsort(-(row_mass * row_share))

                for n_rows in (18, 20, 24, 28, 32, 36):
                    if n_rows > reduced_cols.shape[0] or n_rows == n_cols:
                        continue

                    row_indices = candidate_order[:n_rows]
                    candidate = reduced_cols[row_indices].astype(float)
                    candidate_rows = reduced_row_labels_2[row_indices]
                    candidate_cols = col_labels[col_indices]

                    tw_matrix, tw_row_labels = spectral_permute(
                        candidate,
                        candidate_rows,
                        mode="tw",
                    )
                    tw_score = diagonal_band_mass(tw_matrix, frac=0.12)
                    original_score = diagonal_band_mass(candidate, frac=0.12)
                    score = (
                        tw_score,
                        tw_score - original_score,
                        candidate.shape[0] * candidate.shape[1],
                    )

                    if best is None or score > best["score"]:
                        best = {
                            "score": score,
                            "matrix": candidate,
                            "row_labels": candidate_rows,
                            "col_labels": candidate_cols,
                            "original_score": original_score,
                            "tw_score": tw_score,
                        }

    if best is None:
        raise RuntimeError("Could not find a satisfactory ACS PUMS submatrix.")

    row_meta = pd.DataFrame(
        {
            "occp": best["row_labels"],
            "weighted_total": best["matrix"].sum(axis=1).astype(int),
            "top2_share": np.round(
                (
                    np.sort(best["matrix"], axis=1)[:, -1]
                    + np.sort(best["matrix"], axis=1)[:, -2]
                )
                / best["matrix"].sum(axis=1),
                4,
            ),
        }
    )
    col_meta = pd.DataFrame(
        {
            "indp": best["col_labels"],
            "weighted_total": best["matrix"].sum(axis=0).astype(int),
        }
    )
    return (
        best["matrix"],
        best["row_labels"],
        best["col_labels"],
        row_meta,
        col_meta,
    )


def main() -> None:
    args = parse_args()
    ensure_directories()
    source_path = download_source(args.refresh)

    person_df = load_person_data(source_path)
    matrix_df = person_df.pivot_table(
        index="OCCP",
        columns="INDP",
        values="PWGTP",
        aggfunc="sum",
        fill_value=0,
    )
    original_matrix, row_labels, col_labels, row_meta, col_meta = find_best_submatrix(
        matrix_df
    )
    tw_matrix, tw_row_labels = spectral_permute(
        original_matrix,
        row_labels,
        mode="tw",
    )
    tw_col_labels = recover_column_labels(
        original_matrix,
        row_labels,
        col_labels,
        tw_matrix,
        tw_row_labels,
    )

    write_matrix_csv_fast(
        PROCESSED_DIR / f"{OUTPUT_STEM}_original_matrix.csv",
        original_matrix,
        row_labels,
        col_labels,
    )
    write_matrix_csv_fast(
        PROCESSED_DIR / f"{OUTPUT_STEM}_tw_matrix.csv",
        tw_matrix,
        tw_row_labels,
        tw_col_labels,
    )
    write_metadata_csv(
        PROCESSED_DIR / f"{OUTPUT_STEM}_row_metadata.csv",
        ("occp", "weighted_total", "top2_share"),
        [tuple(row) for row in row_meta.itertuples(index=False, name=None)],
    )
    write_metadata_csv(
        PROCESSED_DIR / f"{OUTPUT_STEM}_col_metadata.csv",
        ("indp", "weighted_total"),
        [tuple(row) for row in col_meta.itertuples(index=False, name=None)],
    )

    figure_path = FIGURES_DIR / f"{OUTPUT_STEM}_original_vs_tw_vs_tw_mosaic.png"
    plot_triptych(
        original_matrix=original_matrix,
        tw_matrix=tw_matrix,
        original_row_labels=row_labels,
        original_col_labels=col_labels,
        tw_row_labels=tw_row_labels,
        tw_col_labels=tw_col_labels,
        output_path=figure_path,
        figure_title="ACS 2023 Massachusetts PUMS",
        subtitle=(
            f"{original_matrix.shape[0]} occupation codes x "
            f"{original_matrix.shape[1]} industry codes, "
            f"weighted by PWGTP"
        ),
        colorbar_label="Weighted person count",
        mask_below=175,
    )

    if args.show:
        image = plt.imread(figure_path)
        plt.figure(figsize=(18, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
