# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
import urllib.request
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

RAW_DIR = REPO_ROOT / "data" / "lodes_rectangular" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "lodes_rectangular" / "processed"
FIGURES_DIR = REPO_ROOT / "data" / "lodes_rectangular" / "figures"

OD_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/tn/od/tn_od_main_JT00_2022.csv.gz"
XWALK_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/tn/tn_xwalk.csv.gz"
OD_FILENAME = "tn_od_main_JT00_2022.csv.gz"
XWALK_FILENAME = "tn_xwalk.csv.gz"
OUTPUT_STEM = "tn_lodes2022_home_county_to_work_county"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Tennessee LODES8 OD data, aggregate to a home-county by "
            "work-county matrix, select a low-entropy commuting submatrix, and "
            "compare original / TW / TW+mosaic views."
        )
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def download_source(url: str, filename: str, refresh: bool) -> Path:
    path = RAW_DIR / filename
    if refresh or not path.exists():
        path.unlink(missing_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_county_matrix(od_path: Path) -> pd.DataFrame:
    od_df = pd.read_csv(od_path, usecols=["w_geocode", "h_geocode", "S000"])
    od_df["w_county"] = od_df["w_geocode"].astype(str).str.zfill(15).str[:5]
    od_df["h_county"] = od_df["h_geocode"].astype(str).str.zfill(15).str[:5]
    county_df = od_df.groupby(["h_county", "w_county"], as_index=False)["S000"].sum()
    return county_df.pivot_table(
        index="h_county",
        columns="w_county",
        values="S000",
        aggfunc="sum",
        fill_value=0,
    )


def load_county_names(path: Path) -> dict[str, str]:
    xwalk = pd.read_csv(path, usecols=["cty", "ctyname"]).drop_duplicates(
        subset=["cty"]
    )
    xwalk["cty"] = xwalk["cty"].astype(str).str.zfill(5)
    return dict(xwalk[["cty", "ctyname"]].itertuples(index=False, name=None))


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

    for min_total in (5000, 10000, 20000, 50000):
        keep_rows = (row_totals >= min_total) & (top2_share >= 0.70)
        if keep_rows.sum() < 8:
            continue

        reduced = matrix[keep_rows]
        reduced_row_labels = row_labels[keep_rows]
        dominant_columns = np.argsort(reduced.sum(axis=0))[::-1]

        for n_cols in (4, 5, 6, 7, 8, 10):
            if n_cols >= reduced.shape[1]:
                continue

            col_indices = dominant_columns[:n_cols]
            candidate_cols = reduced[:, col_indices]
            row_mass = candidate_cols.sum(axis=1)
            row_sorted = np.sort(candidate_cols, axis=1)
            row_share = (row_sorted[:, -1] + row_sorted[:, -2]) / row_mass
            candidate_order = np.argsort(-(row_mass * row_share))

            for n_rows in (10, 12, 15, 18, 20, 24, 28):
                if n_rows > candidate_cols.shape[0] or n_rows == n_cols:
                    continue

                row_indices = candidate_order[:n_rows]
                candidate = candidate_cols[row_indices].astype(float)
                candidate_rows = reduced_row_labels[row_indices]
                candidate_col_labels = col_labels[col_indices]

                tw_matrix, tw_row_labels = spectral_permute(
                    candidate,
                    candidate_rows,
                    mode="tw",
                )
                tw_score = diagonal_band_mass(tw_matrix, frac=0.18)
                original_score = diagonal_band_mass(candidate, frac=0.18)
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
                        "col_labels": candidate_col_labels,
                    }

    if best is None:
        raise RuntimeError("Could not find a satisfactory LODES submatrix.")

    row_meta = pd.DataFrame(
        {
            "home_county": best["row_labels"],
            "commuter_total": best["matrix"].sum(axis=1).astype(int),
        }
    )
    col_meta = pd.DataFrame(
        {
            "work_county": best["col_labels"],
            "job_total": best["matrix"].sum(axis=0).astype(int),
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
    od_path = download_source(OD_URL, OD_FILENAME, args.refresh)
    xwalk_path = download_source(XWALK_URL, XWALK_FILENAME, args.refresh)

    matrix_df = load_county_matrix(od_path)
    county_names = load_county_names(xwalk_path)
    original_matrix, row_codes, col_codes, row_meta, col_meta = find_best_submatrix(
        matrix_df
    )
    row_labels = np.array(
        [county_names.get(code, code).replace(", TN", "") for code in row_codes]
    )
    col_labels = np.array(
        [county_names.get(code, code).replace(", TN", "") for code in col_codes]
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
        ("home_county_code", "home_county_name", "commuter_total"),
        [
            (
                code,
                county_names.get(code, code),
                int(total),
            )
            for code, total in zip(
                row_meta["home_county"],
                row_meta["commuter_total"],
                strict=True,
            )
        ],
    )
    write_metadata_csv(
        PROCESSED_DIR / f"{OUTPUT_STEM}_col_metadata.csv",
        ("work_county_code", "work_county_name", "job_total"),
        [
            (
                code,
                county_names.get(code, code),
                int(total),
            )
            for code, total in zip(
                col_meta["work_county"],
                col_meta["job_total"],
                strict=True,
            )
        ],
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
        figure_title="Tennessee LODES8 2022 Home County to Work County",
        subtitle=(
            f"{original_matrix.shape[0]} home counties x "
            f"{original_matrix.shape[1]} work counties, "
            "S000 commuter counts"
        ),
        colorbar_label="Commuter count (S000)",
        band_frac=0.18,
        mask_below=500,
    )

    if args.show:
        image = plt.imread(figure_path)
        plt.figure(figsize=(18, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
