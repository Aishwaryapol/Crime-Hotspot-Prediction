"""create_ml_features.py

Build a WEEK+LAT_GRID+LON_GRID level ML dataset with lagged historical
features (previous weeks only). This script is safe to run manually
from the VS Code terminal. It does NOT execute automatically here.

Outputs (when you run it):
- data/processed/ml_dataset.csv       : final aggregated dataset including target
- data/processed/X_train.csv, X_test.csv
- data/processed/y_train.csv, y_test.csv
- data/processed/feature_info.txt

Notes:
- Features exclude current-week `CRIME_COUNT` and `Hotspot` to avoid leakage.
- Lagged features are computed using only prior weeks (shift).
"""

import pandas as pd
from pathlib import Path


# File paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "crime_hotspot.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _mode_or_first(series):
    """Return the mode if available, otherwise the first element."""
    m = series.mode()
    if not m.empty:
        return m.iloc[0]
    return series.iloc[0]


def main():
    """Create aggregated ML features at WEEK_START + LAT_GRID + LON_GRID level.

    This script is intended to be run manually by the user from the
    terminal. It validates consistency of `CRIME_COUNT` and `Hotspot`
    within each weekly grid cell before aggregating.
    """

    # --- Load
    print("Loading:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    source_rows = len(df)

    # --- Basic column checks
    required_cols = {"WEEK", "LAT_GRID", "LON_GRID", "CRIME_COUNT", "Hotspot"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in crime_hotspot.csv: {sorted(missing)}")

    # --- Parse WEEK into WEEK_START (week-start date from string like YYYY-MM-DD/...) ---
    df["WEEK_START"] = pd.to_datetime(df["WEEK"].astype(str).str.split("/").str[0], format="%Y-%m-%d", errors="coerce")
    if df["WEEK_START"].isna().any():
        bad = df[df["WEEK_START"].isna()].head(5)
        raise ValueError(f"Some WEEK values could not be parsed into WEEK_START. Examples:\n{bad[["WEEK"]].to_string(index=False)}")

    # --- Consistency check for CRIME_COUNT and Hotspot within each WEEK_START + LAT_GRID + LON_GRID ---
    grp = df.groupby(["WEEK_START", "LAT_GRID", "LON_GRID"], dropna=False)

    # Check CRIME_COUNT consistency
    crime_count_nunique = grp["CRIME_COUNT"].nunique()
    inconsistent_crime = crime_count_nunique[crime_count_nunique > 1]
    if not inconsistent_crime.empty:
        sample = inconsistent_crime.head(10)
        raise ValueError(
            "Inconsistent CRIME_COUNT within WEEK_START+LAT_GRID+LON_GRID groups.\n"
            f"Number of inconsistent groups: {len(inconsistent_crime)}. Examples:\n{sample.to_string()}"
        )

    # Check Hotspot consistency
    hotspot_nunique = grp["Hotspot"].nunique()
    inconsistent_hotspot = hotspot_nunique[hotspot_nunique > 1]
    if not inconsistent_hotspot.empty:
        sample = inconsistent_hotspot.head(10)
        raise ValueError(
            "Inconsistent Hotspot within WEEK_START+LAT_GRID+LON_GRID groups.\n"
            f"Number of inconsistent groups: {len(inconsistent_hotspot)}. Examples:\n{sample.to_string()}"
        )

    # --- Aggregation per requirements ---
    # YEAR -> first, MONTH -> first, DAY_OF_WEEK -> mode, HOUR -> mode,
    # AREA -> mode, AREA NAME -> mode, Crm Cd Desc -> mode, Premis Desc -> mode,
    # CRIME_COUNT -> first, Hotspot -> first
    agg = (
        df
        .groupby(["WEEK_START", "LAT_GRID", "LON_GRID"], as_index=False)
        .agg(
            {
                "YEAR": "first",
                "MONTH": "first",
                "DAY_OF_WEEK": lambda x: _mode_or_first(x),
                "HOUR": lambda x: _mode_or_first(x),
                "AREA": lambda x: _mode_or_first(x),
                "AREA NAME": lambda x: _mode_or_first(x),
                "Crm Cd Desc": lambda x: _mode_or_first(x),
                "Premis Desc": lambda x: _mode_or_first(x),
                "CRIME_COUNT": "first",
                "Hotspot": "first",
            }
        )
    )

    # --- Sort by LAT_GRID, LON_GRID, WEEK_START as requested ---
    agg = agg.sort_values(by=["LAT_GRID", "LON_GRID", "WEEK_START"]).reset_index(drop=True)

    # --- Create historical features using groupby + shift ---
    by_grid = ["LAT_GRID", "LON_GRID"]

    # Ensure operations preserve index alignment
    agg["PREV_WEEK_CRIME_COUNT"] = agg.groupby(by_grid)["CRIME_COUNT"].shift(1)
    agg["PREV_2_WEEK_CRIME_COUNT"] = agg.groupby(by_grid)["CRIME_COUNT"].shift(2)
    agg["PREV_3_WEEK_CRIME_COUNT"] = agg.groupby(by_grid)["CRIME_COUNT"].shift(3)

    # ROLLING_4_WEEK_CRIME_COUNT excluding current week (shift first, then rolling)
    agg["ROLLING_4_WEEK_CRIME_COUNT"] = (
        agg.groupby(by_grid)["CRIME_COUNT"].transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).sum())
    )

    # GRID_TOTAL_PREVIOUS_CRIMES: cumulative previous crimes only (exclude current week)
    agg["GRID_TOTAL_PREVIOUS_CRIMES"] = (
        agg.groupby(by_grid)["CRIME_COUNT"].transform(lambda s: s.shift(1).cumsum())
    )

    # Validation: ensure new columns align with frame index
    assert len(agg["ROLLING_4_WEEK_CRIME_COUNT"]) == len(agg)
    assert len(agg["GRID_TOTAL_PREVIOUS_CRIMES"]) == len(agg)

    # --- Drop rows missing required historical features ---
    required = [
        "PREV_WEEK_CRIME_COUNT",
        "PREV_2_WEEK_CRIME_COUNT",
        "PREV_3_WEEK_CRIME_COUNT",
        "ROLLING_4_WEEK_CRIME_COUNT",
        "GRID_TOTAL_PREVIOUS_CRIMES",
    ]
    initial_agg_rows = len(agg)
    ml_df = agg.dropna(subset=required).reset_index(drop=True)
    removed_rows = initial_agg_rows - len(ml_df)

    # --- Rename text columns ---
    ml_df = ml_df.rename(columns={"Crm Cd Desc": "DOMINANT_CRIME_TYPE", "Premis Desc": "DOMINANT_PREMISE"})

    # --- Final feature columns ---
    feature_columns = [
        "YEAR",
        "MONTH",
        "DAY_OF_WEEK",
        "HOUR",
        "AREA",
        "AREA NAME",
        "LAT_GRID",
        "LON_GRID",
        "DOMINANT_CRIME_TYPE",
        "DOMINANT_PREMISE",
        "PREV_WEEK_CRIME_COUNT",
        "PREV_2_WEEK_CRIME_COUNT",
        "PREV_3_WEEK_CRIME_COUNT",
        "ROLLING_4_WEEK_CRIME_COUNT",
        "GRID_TOTAL_PREVIOUS_CRIMES",
    ]

    # --- Ensure forbidden columns are not in feature list ---
    assert "CRIME_COUNT" not in feature_columns
    assert "Hotspot" not in feature_columns
    assert "WEEK" not in feature_columns
    assert "WEEK_START" not in feature_columns

    # --- Chronological split by YEAR (no shuffling) ---
    train_df = ml_df[ml_df["YEAR"] < 2024].copy()
    test_df = ml_df[ml_df["YEAR"] >= 2024].copy()

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df[["Hotspot"]].copy()
    y_test = test_df[["Hotspot"]].copy()

    # --- Additional assertions before saving ---
    assert len(X_train) + len(X_test) == len(ml_df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # --- Save outputs ---
    ml_df_out = PROCESSED_DIR / "ml_dataset.csv"
    X_train_out = PROCESSED_DIR / "X_train.csv"
    X_test_out = PROCESSED_DIR / "X_test.csv"
    y_train_out = PROCESSED_DIR / "y_train.csv"
    y_test_out = PROCESSED_DIR / "y_test.csv"
    feature_info_out = PROCESSED_DIR / "feature_info.txt"

    ml_df.to_csv(ml_df_out, index=False)
    X_train.to_csv(X_train_out, index=False)
    X_test.to_csv(X_test_out, index=False)
    y_train.to_csv(y_train_out, index=False)
    y_test.to_csv(y_test_out, index=False)

    with feature_info_out.open("w", encoding="utf-8") as fh:
        fh.write("Feature columns used for ML input:\n")
        for c in feature_columns:
            fh.write(f"{c}\n")

    # --- Diagnostics ---
    print("Source rows:", source_rows)
    print("Aggregated weekly grid-cell rows:", initial_agg_rows)
    print("Final ML rows:", len(ml_df))
    print("Rows removed:", removed_rows)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Training Hotspot counts:\n", y_train["Hotspot"].value_counts().to_string())
    print("Testing Hotspot counts:\n", y_test["Hotspot"].value_counts().to_string())
    print("Feature columns:\n", " , ".join(feature_columns))
    if len(train_df):
        print("Training YEAR range:", int(train_df["YEAR"].min()), "to", int(train_df["YEAR"].max()))
    else:
        print("Training YEAR range: none (no training rows)")
    if len(test_df):
        print("Testing YEAR range:", int(test_df["YEAR"].min()), "to", int(test_df["YEAR"].max()))
    else:
        print("Testing YEAR range: none (no testing rows)")


if __name__ == "__main__":
    main()
