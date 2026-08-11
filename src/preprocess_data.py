import pandas as pd
from pathlib import Path


# Define the input and output paths
project_root = Path(__file__).resolve().parent.parent
input_path = project_root / "data" / "raw" / "Crime_Data_from_2020_to_Present.csv"
output_path = project_root / "data" / "processed" / "crime_cleaned.csv"


# Load the original CSV without changing the raw file
print("Loading original dataset...")
df = pd.read_csv(input_path)

# Keep track of the original shape before cleaning
original_shape = df.shape


# Convert DATE OCC to datetime using the known format
print("\nConverting DATE OCC to datetime...")
df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")


# Create a small helper function to convert TIME OCC into an hour value
print("\nExtracting time-based features...")

def parse_time_occ(value):
    """Convert HHMM values like 30, 530, 1230, and 2359 into hour values."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return pd.NA

    if len(digits) < 4:
        digits = digits.zfill(4)

    hour = int(digits[:2])
    return hour if 0 <= hour <= 23 else pd.NA


# Extract date features from DATE OCC
parsed_dates = df["DATE OCC"]

# Create the working dataframe with the requested columns
print("\nBuilding a clean working dataframe...")
working_df = pd.DataFrame({
    "DATE OCC": parsed_dates,
    "YEAR": parsed_dates.dt.year,
    "MONTH": parsed_dates.dt.month,
    "DAY": parsed_dates.dt.day,
    "DAY_OF_WEEK": parsed_dates.dt.day_name(),
    "HOUR": df["TIME OCC"].apply(parse_time_occ),
    "AREA": pd.to_numeric(df["AREA"], errors="coerce"),
    "AREA NAME": df["AREA NAME"],
    "Crm Cd": pd.to_numeric(df["Crm Cd"], errors="coerce"),
    "Crm Cd Desc": df["Crm Cd Desc"],
    "Premis Desc": df["Premis Desc"],
    "LAT": pd.to_numeric(df["LAT"], errors="coerce"),
    "LON": pd.to_numeric(df["LON"], errors="coerce"),
})


# Use DATE OCC to derive the hour when possible, but only if it is valid
print("\nChecking hour values from DATE OCC...")
valid_time_hour = working_df["HOUR"].notna() & working_df["HOUR"].between(0, 23)
valid_date_hour = parsed_dates.notna() & parsed_dates.dt.hour.between(0, 23)

working_df.loc[~valid_time_hour & valid_date_hour, "HOUR"] = parsed_dates[~valid_time_hour & valid_date_hour].dt.hour
working_df["HOUR"] = working_df["HOUR"].fillna(-1).astype(int)


# Remove rows with invalid latitude or longitude values
print("\nRemoving rows with invalid coordinates...")
invalid_coordinate_mask = (
    working_df["LAT"].isna()
    | working_df["LON"].isna()
    | working_df["LAT"].eq(0)
    | working_df["LON"].eq(0)
)
cleaned_df = working_df.loc[~invalid_coordinate_mask].copy()


# Fill missing categorical values sensibly without deleting rows
print("\nHandling missing categorical values...")
cleaned_df["AREA NAME"] = cleaned_df["AREA NAME"].fillna("Unknown")
cleaned_df["Crm Cd Desc"] = cleaned_df["Crm Cd Desc"].fillna("Unknown")
cleaned_df["Premis Desc"] = cleaned_df["Premis Desc"].fillna("Unknown")
cleaned_df["AREA"] = cleaned_df["AREA"].fillna(-1).astype(int)
cleaned_df["Crm Cd"] = cleaned_df["Crm Cd"].fillna(-1).astype(int)


# Save the cleaned dataframe to the processed folder
output_path.parent.mkdir(parents=True, exist_ok=True)
cleaned_df.to_csv(output_path, index=False)
print(f"\nSaved cleaned dataset to: {output_path}")


# Print summary information for the beginner-friendly inspection step
print("\nOriginal shape:", original_shape)
print("Cleaned shape:", cleaned_df.shape)
print("Rows removed:", original_shape[0] - cleaned_df.shape[0])
print("Remaining missing values:")
print(cleaned_df.isnull().sum())
print("Minimum latitude:", cleaned_df["LAT"].min())
print("Maximum latitude:", cleaned_df["LAT"].max())
print("Minimum longitude:", cleaned_df["LON"].min())
print("Maximum longitude:", cleaned_df["LON"].max())
print("Number of unique AREA NAME values:", cleaned_df["AREA NAME"].nunique())
print("Number of unique crime types:", cleaned_df["Crm Cd Desc"].nunique())
print("Hour distribution summary:")
print(cleaned_df["HOUR"].value_counts().sort_index())
