import pandas as pd
from pathlib import Path

# Define file paths
project_root = Path(__file__).resolve().parent.parent
input_path = project_root / "data" / "processed" / "crime_hotspot.csv"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Load the hotspot dataset
print("Loading crime_hotspot.csv...")
loaded_df = pd.read_csv(input_path)

# Print the source dataset diagnostics before preparation
print("\nML source dataset:", "data/processed/crime_hotspot.csv")
print("Full loaded shape:", loaded_df.shape)
print(loaded_df["Hotspot"].value_counts())
print((loaded_df["Hotspot"].value_counts(normalize=True) * 100).round(2))

# Deduplicate the data to keep only weekly geographic grid-cell records
print("\nDeduplicating to weekly grid-cell level...")
df = loaded_df.drop_duplicates(subset=["WEEK", "LAT_GRID", "LON_GRID"]).copy()
print("Deduplicated shape:", df.shape)
print(df["Hotspot"].value_counts())
print((df["Hotspot"].value_counts(normalize=True) * 100).round(2))

# Target column
target_column = "Hotspot"

# Features to use for ML input
feature_columns = [
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
    "LAT_GRID",
    "LON_GRID",
]

# Check required columns are present
required_columns = feature_columns + [target_column, "WEEK"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in dataset: {missing_columns}")

# Parse the WEEK field to sort chronologically
print("Parsing WEEK column for chronological order...")
df["WEEK_START"] = pd.to_datetime(df["WEEK"].astype(str).str.split("/").str[0], format="%Y-%m-%d", errors="coerce")

# Drop rows with missing essential values only when necessary
print("Dropping rows with missing values in selected features or target...")
keep_columns = feature_columns + [target_column, "WEEK", "WEEK_START"]
initial_count = len(df)
df = df[keep_columns].dropna()
final_count = len(df)
print(f"Rows removed due to missing values: {initial_count - final_count}")

# Sort records chronologically by WEEK_START, then by YEAR and month/day
print("Sorting data chronologically by WEEK...")
df = df.sort_values(by=["WEEK_START", "YEAR", "MONTH", "DAY", "HOUR"]).reset_index(drop=True)

# Print final ML source diagnostics after sorting
print("\nFinal ML dataframe shape:", df.shape)
print(df["Hotspot"].value_counts())
print((df["Hotspot"].value_counts(normalize=True) * 100).round(2))

# Create train/test split by year
print("Splitting data into training and testing sets...")
train_df = df[df["YEAR"] < 2024].copy()
test_df = df[df["YEAR"] >= 2024].copy()

# Verify chronological order between train and test
train_start = train_df["WEEK_START"].min()
train_end = train_df["WEEK_START"].max()
test_start = test_df["WEEK_START"].min()
test_end = test_df["WEEK_START"].max()

if pd.notna(test_start) and pd.notna(train_end) and test_start < train_end:
    raise ValueError("Test data contains weeks before the end of the training period.")

# Prepare X and y datasets
X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()
y_train = train_df[[target_column]].copy()
y_test = test_df[[target_column]].copy()

# Save datasets to CSV
print("Saving prepared datasets...")
X_train.to_csv(processed_dir / "X_train.csv", index=False)
X_test.to_csv(processed_dir / "X_test.csv", index=False)
y_train.to_csv(processed_dir / "y_train.csv", index=False)
y_test.to_csv(processed_dir / "y_test.csv", index=False)

# Save feature names to a simple text file
feature_info_path = processed_dir / "feature_info.txt"
with feature_info_path.open("w", encoding="utf-8") as f:
    f.write("Feature columns used for ML input:\n")
    for feature in feature_columns:
        f.write(f"{feature}\n")

# Print summary information
print("\nML dataset preparation summary:")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Hotspot=1 in training: {int((y_train[target_column] == 1).sum())}")
print(f"Hotspot=0 in training: {int((y_train[target_column] == 0).sum())}")
print(f"Hotspot=1 in testing: {int((y_test[target_column] == 1).sum())}")
print(f"Hotspot=0 in testing: {int((y_test[target_column] == 0).sum())}")
print("\nFeature column names:")
for feature in feature_columns:
    print(feature)
print("\nTraining YEAR range:")
print(f"Min: {train_df['YEAR'].min()}")
print(f"Max: {train_df['YEAR'].max()}")
print("\nTesting YEAR range:")
print(f"Min: {test_df['YEAR'].min()}")
print(f"Max: {test_df['YEAR'].max()}")
print("\nTraining DATE OCC range:")
print(f"Min: {train_df['WEEK_START'].min()}")
print(f"Max: {train_df['WEEK_START'].max()}")
print("\nTesting DATE OCC range:")
print(f"Min: {test_df['WEEK_START'].min()}")
print(f"Max: {test_df['WEEK_START'].max()}")
print("\nSaved feature names to:", feature_info_path)
