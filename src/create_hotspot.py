import numpy as np
import pandas as pd
from pathlib import Path

# Define file paths
project_root = Path(__file__).resolve().parent.parent
input_path = project_root / "data" / "processed" / "crime_cleaned.csv"
output_path = project_root / "data" / "processed" / "crime_hotspot.csv"

# Load the cleaned dataset
print("Loading cleaned dataset...")
df = pd.read_csv(input_path)

# Convert DATE OCC to datetime for weekly grouping
print("Converting DATE OCC to datetime...")
df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")

# Create a weekly period column for each crime
print("Creating weekly period column...")
df["WEEK"] = df["DATE OCC"].dt.to_period("W").astype(str)

# Create geographic grid cells using a simple 0.01-degree grid
print("Creating geographic grid cells...")
df["LAT_GRID"] = np.floor(df["LAT"] / 0.01) * 0.01
df["LON_GRID"] = np.floor(df["LON"] / 0.01) * 0.01

# Round grid cell labels to keep the values clean
df["LAT_GRID"] = df["LAT_GRID"].round(4)
df["LON_GRID"] = df["LON_GRID"].round(4)

# Count crimes for each weekly grid cell
print("Counting crimes per weekly grid cell...")
grid_counts = (
    df.groupby(["WEEK", "LAT_GRID", "LON_GRID"])
    .size()
    .reset_index(name="CRIME_COUNT")
)

# Determine the hotspot threshold using the 75th percentile
print("Calculating hotspot threshold...")
hotspot_threshold = grid_counts["CRIME_COUNT"].quantile(0.75)

# Create the Hotspot target based on weekly grid-cell crime density
grid_counts["Hotspot"] = (grid_counts["CRIME_COUNT"] >= hotspot_threshold).astype(int)

# Merge hotspot labels back into the main dataframe
print("Merging hotspot labels into the dataset...")
df = df.merge(
    grid_counts,
    on=["WEEK", "LAT_GRID", "LON_GRID"],
    how="left",
)

# Select the requested columns for the output file
output_columns = [
    "DATE OCC",
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "HOUR",
    "AREA",
    "AREA NAME",
    "Crm Cd",
    "Crm Cd Desc",
    "Premis Desc",
    "LAT",
    "LON",
    "WEEK",
    "LAT_GRID",
    "LON_GRID",
    "CRIME_COUNT",
    "Hotspot",
]

result_df = df[output_columns].copy()

# Save the hotspot dataset to the processed folder
output_path.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(output_path, index=False)
print(f"Saved hotspot dataset to: {output_path}")

# Print summary information
weekly_grid_records = len(grid_counts)
hotspot_count = int(grid_counts["Hotspot"].sum())
non_hotspot_count = int((grid_counts["Hotspot"] == 0).sum())
unique_grid_cells = grid_counts[["LAT_GRID", "LON_GRID"]].drop_duplicates().shape[0]
unique_weeks = grid_counts["WEEK"].nunique()

print("\nHotspot generation summary:")
print("Number of weekly grid-cell records:", weekly_grid_records)
print("Hotspot threshold (75th percentile):", hotspot_threshold)
print("Number of Hotspot = 1 records:", hotspot_count)
print("Number of Hotspot = 0 records:", non_hotspot_count)
print("Percentage of hotspots:", f"{hotspot_count / weekly_grid_records * 100:.2f}%")
print("Number of unique geographic grid cells:", unique_grid_cells)
print("Number of unique weeks:", unique_weeks)
