"""Inspect and summarize Random Forest feature importances.

This script performs validation on the CSV exported by the Random Forest
training step and produces a top-20 CSV and a horizontal bar chart PNG.

Do NOT train or load the model here. Do NOT modify datasets.

Run manually from project root (example):
    python src/analyze_feature_importance.py

Beginners: this file uses `pathlib` for paths and `matplotlib` for plotting.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    # Resolve project root and important paths using pathlib
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    viz_dir = project_root / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    fi_path = models_dir / "random_forest_feature_importance.csv"
    top20_out = models_dir / "top_20_feature_importance.csv"
    viz_out = viz_dir / "random_forest_top_20_features.png"

    # 1) Load the CSV
    if not fi_path.exists():
        raise FileNotFoundError(f"Expected feature importance CSV not found: {fi_path}")

    df = pd.read_csv(fi_path)

    # 2) Verify required columns
    expected_cols = {"feature", "importance"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {fi_path}: {missing_cols}")

    # 3) Verify importance values are numeric (coerce then check for NaNs)
    # Keep original series for user-friendly printing
    importance = pd.to_numeric(df["importance"], errors="coerce")
    if importance.isna().any():
        bad_idx = importance[importance.isna()].index.tolist()
        raise ValueError(f"Non-numeric importance values found at rows: {bad_idx}")

    # 4) Verify no missing feature names or importance values
    # First detect true nulls in the original column (NaN)
    if df["feature"].isna().any():
        bad_idx = df[df["feature"].isna()].index.tolist()
        raise ValueError(f"Missing (NaN) feature names at rows: {bad_idx}")

    # Then ensure there are no empty strings once converted to stripped strings
    feature_names = df["feature"].astype(str).str.strip()
    if (feature_names == "").any():
        bad_idx = feature_names[feature_names == ""].index.tolist()
        raise ValueError(f"Empty feature names found at rows: {bad_idx}")

    # Build a clean DataFrame with validated types
    clean_df = pd.DataFrame({"feature": feature_names, "importance": importance})

    # 5) Sort features by importance descending
    clean_df = clean_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    # 6) Print the TOP 20 most important transformed features
    top_n = 20
    top_df = clean_df.head(top_n)

    print("Top {} transformed features (most -> least important):".format(top_n))
    # Nicely print name and importance with formatting
    for i, row in top_df.iterrows():
        print(f"{i+1:2d}. {row['feature']} — importance={row['importance']:.6f}")

    # 7) Print summary statistics
    total_features = len(clean_df)
    importance_sum = float(clean_df["importance"].sum())
    highest = float(clean_df["importance"].max())
    lowest = float(clean_df["importance"].min())

    print("")
    print(f"Total transformed features: {total_features}")
    print(f"Sum of importances: {importance_sum:.6f}")
    print(f"Highest importance: {highest:.6f}")
    print(f"Lowest importance: {lowest:.6f}")

    # 8) Save the sorted top 20 results to models/top_20_feature_importance.csv
    top_df.to_csv(top20_out, index=False)
    print(f"Saved top {top_n} importances to: {top20_out}")

    # 9) Create a simple horizontal bar chart of the top 20 features using matplotlib
    # We reverse the DataFrame so the largest importance appears at the top of the chart
    plot_df = top_df.iloc[::-1]
    plt.figure(figsize=(10, 8))
    bars = plt.barh(plot_df["feature"], plot_df["importance"], color="#2b8cbe")
    plt.xlabel("Importance")
    plt.title("Random Forest Top 20 Feature Importances")
    plt.tight_layout()
    # Annotate bars with importance values for readability
    for bar in bars:
        width = bar.get_width()
        plt.gca().text(width + (importance_sum * 0.001), bar.get_y() + bar.get_height() / 2,
                       f"{width:.4f}", va="center", fontsize=8)

    plt.savefig(viz_out, dpi=150)
    plt.close()
    print(f"Saved visualization to: {viz_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
