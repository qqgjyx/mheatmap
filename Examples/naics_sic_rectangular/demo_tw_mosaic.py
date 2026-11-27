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

import numpy as np
import pandas as pd
from mheatmap.graph import spectral_permute

from _rectangular_tw_common import (
    plot_triptych,
    recover_column_labels,
    write_matrix_csv_fast,
    write_metadata_csv,
)

RAW_DIR = REPO_ROOT / "data" / "naics_sic_rectangular" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "naics_sic_rectangular" / "processed"
FIGURES_DIR = REPO_ROOT / "data" / "naics_sic_rectangular" / "figures"

SOURCE_URL = (
    "https://www2.census.gov/library/reference/naics/technical-documentation/"
    "concordance/1987_sic_to_2002_naics.xls"
)
RAW_XLS_FILENAME = "1987_sic_to_2002_naics.xls"
RAW_CSV_FILENAME = "1987_sic_to_2002_naics.csv"
OUTPUT_STEM = "sic1987_to_naics2002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official 1987 SIC -> 2002 NAICS concordance, "
            "construct the full binary rectangular mapping matrix, and compare "
            "original / TW / TW+mosaic views."
        )
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def download_source(refresh: bool) -> Path:
    path = RAW_DIR / RAW_XLS_FILENAME
    if refresh or not path.exists():
        path.unlink(missing_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)
    return path


def ensure_csv_source(xls_path: Path, refresh: bool) -> Path:
    csv_path = RAW_DIR / RAW_CSV_FILENAME
    if csv_path.exists() and not refresh:
        return csv_path

    try:
        df = pd.read_excel(xls_path)
    except ImportError:
        if csv_path.exists():
            return csv_path
        raise RuntimeError(
            "This example needs either xlrd to read the downloaded .xls file or an "
            "already-generated CSV cache at "
            f"{csv_path}."
        ) from None

    df.to_csv(csv_path, index=False)
    return csv_path


def build_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    df["SIC"] = df["SIC"].astype(str).str.extract(r"(\d+)")[0].str.zfill(4)
    df["2002 NAICS"] = df["2002 NAICS"].astype(str).str.extract(r"(\d+)")[0]
    df = df.dropna(subset=["SIC", "2002 NAICS"]).copy()
    df = df.drop_duplicates(subset=["SIC", "2002 NAICS"])

    sic_title_col = "SIC Title (and note)"

    sic_meta = (
        df[["SIC", sic_title_col]]
        .drop_duplicates(subset=["SIC"])
        .sort_values("SIC")
        .reset_index(drop=True)
    )
    naics_meta = (
        df[["2002 NAICS", "2002 NAICS Title"]]
        .drop_duplicates(subset=["2002 NAICS"])
        .sort_values("2002 NAICS")
        .reset_index(drop=True)
    )

    row_labels = sic_meta["SIC"].to_numpy()
    col_labels = naics_meta["2002 NAICS"].to_numpy()
    row_lookup = {label: index for index, label in enumerate(row_labels)}
    col_lookup = {label: index for index, label in enumerate(col_labels)}

    matrix = np.zeros((len(row_labels), len(col_labels)), dtype=float)
    for sic, naics in df[["SIC", "2002 NAICS"]].itertuples(index=False):
        matrix[row_lookup[sic], col_lookup[naics]] = 1.0

    return matrix, row_labels, col_labels, df


def main() -> None:
    args = parse_args()
    ensure_directories()
    source_path = download_source(args.refresh)
    csv_path = ensure_csv_source(source_path, args.refresh)

    original_matrix, row_labels, col_labels, mapping_df = build_matrix(csv_path)
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
        ("sic", "sic_title"),
        [
            (code, title)
            for code, title in zip(
                mapping_df[["SIC", "SIC Title (and note)"]]
                .drop_duplicates(subset=["SIC"])
                .sort_values("SIC")["SIC"],
                mapping_df[["SIC", "SIC Title (and note)"]]
                .drop_duplicates(subset=["SIC"])
                .sort_values("SIC")["SIC Title (and note)"],
                strict=True,
            )
        ],
    )
    write_metadata_csv(
        PROCESSED_DIR / f"{OUTPUT_STEM}_col_metadata.csv",
        ("naics_2002", "naics_2002_title"),
        [
            (code, title)
            for code, title in zip(
                mapping_df[["2002 NAICS", "2002 NAICS Title"]]
                .drop_duplicates(subset=["2002 NAICS"])
                .sort_values("2002 NAICS")["2002 NAICS"],
                mapping_df[["2002 NAICS", "2002 NAICS Title"]]
                .drop_duplicates(subset=["2002 NAICS"])
                .sort_values("2002 NAICS")["2002 NAICS Title"],
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
        figure_title="1987 SIC to 2002 NAICS Concordance",
        subtitle=(
            f"{original_matrix.shape[0]} SIC rows x "
            f"{original_matrix.shape[1]} NAICS columns, "
            f"{int(original_matrix.sum())} official mapping pairs"
        ),
        colorbar_label="Binary concordance entry",
    )

    if args.show:
        import matplotlib.pyplot as plt

        image = plt.imread(figure_path)
        plt.figure(figsize=(18, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
