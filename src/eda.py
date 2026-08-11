import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set a clean seaborn style for plots
sns.set(style="whitegrid")

# Paths
project_root = Path(__file__).resolve().parent.parent
input_path = project_root / "data" / "processed" / "crime_hotspot.csv"
visualizations_path = project_root / "visualizations"
visualizations_path.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading hotspot dataset for EDA...")
loaded_df = pd.read_csv(input_path)

# Convert DATE OCC to datetime for plotting if needed
print("Converting DATE OCC to datetime for EDA...")
loaded_df["DATE OCC"] = pd.to_datetime(loaded_df["DATE OCC"], errors="coerce")

# Use only the weekly grid-cell dataset by removing duplicate WEEK/LAT_GRID/LON_GRID rows
print("\nChecking dataset level and removing duplicate grid-cell records if needed...")
df = loaded_df.drop_duplicates(subset=["WEEK", "LAT_GRID", "LON_GRID"]).copy()

# Print dataset diagnostics before visualizations
print("\nDataset diagnostics:")
print("df.shape:", df.shape)
print(df["Hotspot"].value_counts())
print((df["Hotspot"].value_counts(normalize=True) * 100).round(2))
print("unique weeks:", df["WEEK"].nunique())
print("unique geographic grid cells:", df[["LAT_GRID", "LON_GRID"]].drop_duplicates().shape[0])

# Print useful findings
print("\nEDA summary findings:")
total_records = len(df)
hotspot_count = int((df["Hotspot"] == 1).sum())
non_hotspot_count = int((df["Hotspot"] == 0).sum())
hotspot_percentage = hotspot_count / total_records * 100
most_common_crime = df["Crm Cd Desc"].mode().iloc[0]
top_10_crime_types = df["Crm Cd Desc"].value_counts().head(10)
day_with_highest_crimes = df["DAY_OF_WEEK"].value_counts().idxmax()
hour_with_highest_crimes = int(df["HOUR"].value_counts().idxmax())
year_with_highest_crimes = int(df["YEAR"].value_counts().idxmax())
unique_areas = df["AREA NAME"].nunique()
unique_grid_cells = df[["LAT_GRID", "LON_GRID"]].drop_duplicates().shape[0]

print(f"Total number of records: {total_records}")
print(f"Number of hotspots: {hotspot_count}")
print(f"Percentage of hotspots: {hotspot_percentage:.2f}%")
print(f"Most common crime type: {most_common_crime}")
print("Top 10 crime types:")
print(top_10_crime_types)
print(f"Day with the highest number of crimes: {day_with_highest_crimes}")
print(f"Hour with the highest number of crimes: {hour_with_highest_crimes}")
print(f"Year with the highest number of crimes: {year_with_highest_crimes}")
print(f"Number of unique areas: {unique_areas}")
print(f"Number of unique geographic grid cells: {unique_grid_cells}")

# Helper to save figures

def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(visualizations_path / filename, dpi=150)
    plt.close(fig)


# 1. Crime count by year
print("\nCreating crime count by year plot...")
year_counts = df["YEAR"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=year_counts.index, y=year_counts.values, palette="viridis", ax=ax)
ax.set_title("Crime Count by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Crimes")
save_fig(fig, "crime_by_year.png")

# 2. Crime count by day of week
print("Creating crime count by day of week plot...")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_counts = df["DAY_OF_WEEK"].value_counts().reindex(weekday_order)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=day_counts.index, y=day_counts.values, palette="magma", ax=ax)
ax.set_title("Crime Count by Day of Week")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Number of Crimes")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
save_fig(fig, "crime_by_day.png")

# 3. Crime count by hour
print("Creating crime count by hour plot...")
hour_counts = df["HOUR"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=hour_counts.index, y=hour_counts.values, palette="coolwarm", ax=ax)
ax.set_title("Crime Count by Hour")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Crimes")
save_fig(fig, "crime_by_hour.png")

# 4. Top 10 crime types
print("Creating top 10 crime types plot...")
top_10 = top_10_crime_types.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_10.values, y=top_10.index, palette="plasma", ax=ax)
ax.set_title("Top 10 Crime Types")
ax.set_xlabel("Number of Crimes")
ax.set_ylabel("Crime Type")
save_fig(fig, "top_10_crime_types.png")

# 5. Hotspot vs non-hotspot distribution
print("Creating hotspot distribution plot...")
hotspot_counts = df["Hotspot"].value_counts().sort_index()
labels = ["Non-hotspot", "Hotspot"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=labels, y=hotspot_counts.values, palette=["#4c72b0", "#dd8452"], ax=ax)
ax.set_title("Hotspot vs Non-hotspot Distribution")
ax.set_xlabel("Hotspot")
ax.set_ylabel("Number of Records")
save_fig(fig, "hotspot_distribution.png")

# 6. Geographic crime distribution
print("Creating geographic crime distribution scatter plot...")
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df["LON_GRID"], df["LAT_GRID"], s=10, alpha=0.5, color="#2ca02c")
ax.set_title("Geographic Crime Distribution by Grid Cell")
ax.set_xlabel("Longitude Grid")
ax.set_ylabel("Latitude Grid")
save_fig(fig, "crime_geographic_distribution.png")

# 7. Geographic hotspot distribution
print("Creating hotspot geographic distribution scatter plot...")
hotspot_df = df[df["Hotspot"] == 1]
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(hotspot_df["LON_GRID"], hotspot_df["LAT_GRID"], s=15, alpha=0.6, color="#d62728")
ax.set_title("Geographic Hotspot Distribution by Grid Cell")
ax.set_xlabel("Longitude Grid")
ax.set_ylabel("Latitude Grid")
save_fig(fig, "hotspot_geographic_distribution.png")

print("\nSaved all requested EDA visualizations to the visualizations/ folder.")
