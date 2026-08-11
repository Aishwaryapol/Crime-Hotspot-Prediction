"""Save prediction history to CSV.

Provides `save_prediction(input_data, prediction_result)` which appends a
single prediction record to `data/predictions/prediction_history.csv`.

This module does NOT load models or preprocessors and does NOT perform
predictions. It only records inputs and prediction outputs for later analysis.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd


# Exact CSV columns required by the project
CSV_COLUMNS = [
    "timestamp",
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR",
    "AREA",
    "AREA_NAME",
    "LAT_GRID",
    "LON_GRID",
    "DOMINANT_CRIME_TYPE",
    "DOMINANT_PREMISE",
    "PREV_WEEK_CRIME_COUNT",
    "PREV_2_WEEK_CRIME_COUNT",
    "PREV_3_WEEK_CRIME_COUNT",
    "ROLLING_4_WEEK_CRIME_COUNT",
    "GRID_TOTAL_PREVIOUS_CRIMES",
    "prediction",
    "prediction_label",
    "probability",
]


def save_prediction(input_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> Path:
    """Append a prediction record to `data/predictions/prediction_history.csv`.

    Parameters
    - input_data: dictionary matching the input accepted by the prediction
      pipeline (keys like 'YEAR', 'AREA NAME', etc.).
    - prediction_result: dictionary returned by `predict_hotspot()`, must
      include keys 'prediction', 'prediction_label', 'probability'.

    Returns the Path to the CSV file written.
    """
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prediction_history.csv"

    # 1) Validate prediction_result contains required keys
    required_pred_keys = {"prediction", "prediction_label", "probability"}
    if not required_pred_keys.issubset(set(prediction_result.keys())):
        missing = required_pred_keys - set(prediction_result.keys())
        raise ValueError(f"prediction_result is missing keys: {sorted(missing)}")

    # 2) Validate probability is numeric and between 0 and 1
    prob = prediction_result["probability"]
    try:
        prob_val = float(prob)
    except Exception:
        raise ValueError(f"probability must be numeric between 0 and 1. Got: {prob!r}")
    if not (0.0 <= prob_val <= 1.0):
        raise ValueError(f"probability must be between 0 and 1. Got: {prob_val}")

    # 3) Validate prediction is 0 or 1
    pred = prediction_result["prediction"]
    if int(pred) not in (0, 1):
        raise ValueError(f"prediction must be 0 or 1. Got: {pred!r}")

    # 4) Validate input_data has required fields. The prediction pipeline uses
    #    'AREA NAME' (with space) but CSV requires 'AREA_NAME' (underscore).
    expected_input_fields = {
        "YEAR",
        "MONTH",
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
        "DAY_OF_WEEK",
    }

    missing_inputs = expected_input_fields - set(input_data.keys())
    extra_inputs = set(input_data.keys()) - expected_input_fields
    if missing_inputs:
        raise ValueError(f"input_data is missing fields: {sorted(missing_inputs)}")
    if extra_inputs:
        raise ValueError(f"input_data contains unexpected fields: {sorted(extra_inputs)}")

    # 5) Build the row to save using exact CSV column names and ordering
    # Map 'AREA NAME' -> 'AREA_NAME'
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "YEAR": input_data["YEAR"],
        "MONTH": input_data["MONTH"],
        "DAY_OF_WEEK": input_data["DAY_OF_WEEK"],
        "HOUR": input_data["HOUR"],
        "AREA": input_data["AREA"],
        "AREA_NAME": input_data["AREA NAME"],
        "LAT_GRID": input_data["LAT_GRID"],
        "LON_GRID": input_data["LON_GRID"],
        "DOMINANT_CRIME_TYPE": input_data["DOMINANT_CRIME_TYPE"],
        "DOMINANT_PREMISE": input_data["DOMINANT_PREMISE"],
        "PREV_WEEK_CRIME_COUNT": input_data["PREV_WEEK_CRIME_COUNT"],
        "PREV_2_WEEK_CRIME_COUNT": input_data["PREV_2_WEEK_CRIME_COUNT"],
        "PREV_3_WEEK_CRIME_COUNT": input_data["PREV_3_WEEK_CRIME_COUNT"],
        "ROLLING_4_WEEK_CRIME_COUNT": input_data["ROLLING_4_WEEK_CRIME_COUNT"],
        "GRID_TOTAL_PREVIOUS_CRIMES": input_data["GRID_TOTAL_PREVIOUS_CRIMES"],
        "prediction": int(prediction_result["prediction"]),
        "prediction_label": prediction_result["prediction_label"],
        "probability": float(prob_val),
    }

    df_row = pd.DataFrame([row], columns=CSV_COLUMNS)

    # 6) Write or append to CSV without duplicating header
    write_header = not out_path.exists()
    # Use mode 'a' to append if file exists
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)

    print("Prediction saved successfully.")
    print("CSV file:")
    print(out_path.as_posix())

    return out_path


if __name__ == "__main__":
    # Small test that does NOT call the ML model. It only tests CSV writing.
    example_input = {
        "YEAR": 2024,
        "MONTH": 12,
        "HOUR": 22,
        "AREA": 1,
        "AREA NAME": "Central",
        "LAT_GRID": 34.05,
        "LON_GRID": -118.25,
        "DOMINANT_CRIME_TYPE": "VEHICLE - STOLEN",
        "DOMINANT_PREMISE": "STREET",
        "PREV_WEEK_CRIME_COUNT": 8,
        "PREV_2_WEEK_CRIME_COUNT": 6,
        "PREV_3_WEEK_CRIME_COUNT": 7,
        "ROLLING_4_WEEK_CRIME_COUNT": 25,
        "GRID_TOTAL_PREVIOUS_CRIMES": 150,
        "DAY_OF_WEEK": "Saturday",
    }

    # Example prediction result (this is a made-up example for storage testing)
    example_result = {
        "prediction": 1,
        "prediction_label": "HOTSPOT",
        "probability": 0.8339,
    }

    print("Saving example prediction to CSV (test only).")
    out = save_prediction(example_input, example_result)

    # Verify CSV exists and print summary info
    df = pd.read_csv(out)
    print("")
    print(f"CSV path: {out.as_posix()}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Last row:")
    print(df.tail(1).to_dict(orient="records")[0])
