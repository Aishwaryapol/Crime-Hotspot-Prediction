import pandas as pd
from pathlib import Path

# Get the project root and point to the raw dataset file
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "Crime_Data_from_2020_to_Present.csv"

# Load the CSV file into a pandas DataFrame
print("Loading dataset...")
df = pd.read_csv(data_path)

# Print the dataset shape
print("\nDataset shape (rows, columns):")
print(df.shape)

# Print all column names
print("\nColumn names:")
for column in df.columns:
    print(column)

# Print the first 5 rows of the dataset
print("\nFirst 5 rows:")
print(df.head())

# Print pandas info() summary
print("\nDataset info:")
df.info()

# Print the data types of all columns
print("\nData types:")
print(df.dtypes)

# Print the number of missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# Print the number of duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Print basic descriptive statistics
print("\nBasic descriptive statistics:")
print(df.describe(include="all"))

# Print the number of unique values for object/string columns
object_columns = df.select_dtypes(include=["object", "string"]).columns
if len(object_columns) > 0:
    print("\nUnique value counts for object/string columns:")
    for column in object_columns:
        print(f"{column}: {df[column].nunique()}")
else:
    print("\nNo object/string columns found.")

# Convert DATE OCC to datetime and report parse issues
print("\nDATE OCC date summary:")
if "DATE OCC" in df.columns:
    parsed_dates = pd.to_datetime(df["DATE OCC"], errors="coerce")
    unparsable_count = parsed_dates.isna().sum()
    valid_dates = parsed_dates.dropna()

    print(f"Number of dates that could not be parsed: {unparsable_count}")
    if not valid_dates.empty:
        print(f"Minimum valid DATE OCC: {valid_dates.min()}")
        print(f"Maximum valid DATE OCC: {valid_dates.max()}")
    else:
        print("No valid DATE OCC dates found.")
else:
    print("DATE OCC column not found in the dataset.")
