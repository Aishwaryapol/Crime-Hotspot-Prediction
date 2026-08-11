"""Prediction helper for Crime Hotspot model.

Loads an already-fitted preprocessor and an already-trained Random Forest model
and exposes a `predict_hotspot(input_data)` function that accepts a single-row
dictionary of features and returns a prediction, label, and probability.

This script is intentionally beginner-friendly and heavily commented.

Do NOT fit or retrain any models here. Do NOT modify datasets.
"""

from pathlib import Path
import sys
from typing import Dict, Any

import joblib
import pandas as pd


# Exact feature list and order expected by the trained model / preprocessor.
FEATURE_COLUMNS = [
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
]

# Numeric features that must be convertible to numbers
NUMERIC_FEATURES = [
    "YEAR",
    "MONTH",
    "HOUR",
    "AREA",
    "LAT_GRID",
    "LON_GRID",
    "PREV_WEEK_CRIME_COUNT",
    "PREV_2_WEEK_CRIME_COUNT",
    "PREV_3_WEEK_CRIME_COUNT",
    "ROLLING_4_WEEK_CRIME_COUNT",
    "GRID_TOTAL_PREVIOUS_CRIMES",
]


def load_artifacts(project_root: Path):
    """Load preprocessor and model from the project's `models/` directory.

    Raises clear exceptions if artifacts are missing or cannot be loaded.
    """
    models_dir = project_root / "models"
    preprocessor_path = models_dir / "preprocessor.joblib"
    model_path = models_dir / "random_forest.joblib"

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception as exc:  # pragma: no cover - display friendly error
        raise RuntimeError(f"Failed to load preprocessor from {preprocessor_path}: {exc}")

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - display friendly error
        raise RuntimeError(f"Failed to load model from {model_path}: {exc}")

    return preprocessor, model


def predict_hotspot(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict whether a grid-week is a hotspot from a single-row dict.

    Parameters
    - input_data: dict with exactly the keys in `FEATURE_COLUMNS`.

    Returns a dictionary with keys: `prediction` (0/1), `prediction_label`, and `probability`.
    """
    project_root = Path(__file__).resolve().parent.parent

    # 1) Validate keys: required and no unexpected keys
    input_keys = set(input_data.keys())
    expected_keys = set(FEATURE_COLUMNS)

    # Explicitly disallow leaking columns
    if "CRIME_COUNT" in input_keys or "Hotspot" in input_keys:
        raise ValueError("Input must not contain 'CRIME_COUNT' or 'Hotspot' fields (target leakage).")

    missing = expected_keys - input_keys
    extra = input_keys - expected_keys
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected features provided: {sorted(extra)}")

    # 2) Convert to one-row DataFrame and ensure column ordering
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    # 3) Validate numeric fields are numeric (attempt conversion)
    for col in NUMERIC_FEATURES:
        try:
            input_df[col] = pd.to_numeric(input_df[col], errors="raise")
        except Exception:
            raise ValueError(f"Feature '{col}' must be numeric. Value: {input_df.loc[0, col]!r}")

    # 4) Load preprocessor and model (do not fit)
    preprocessor, model = load_artifacts(project_root)

    # 5) Transform using pre-fitted preprocessor (do not call fit/fit_transform)
    try:
        X_transformed = preprocessor.transform(input_df)
    except Exception as exc:
        raise RuntimeError(f"Preprocessor.transform failed: {exc}")

    # 6) Predict and get probability for positive class (1)
    try:
        pred = model.predict(X_transformed)
    except Exception as exc:
        raise RuntimeError(f"Model.predict failed: {exc}")

    try:
        proba_all = model.predict_proba(X_transformed)
        # Probability for class 1; handle case if classes are ordered differently
        # We assume binary classification with classes [0,1]
        if proba_all.shape[1] == 1:
            # Some models may return a single-prob column; interpret as probability for class 1
            prob_pos = float(proba_all.ravel()[0])
        else:
            prob_pos = float(proba_all[:, 1].ravel()[0])
    except Exception as exc:
        raise RuntimeError(f"Model.predict_proba failed: {exc}")

    prediction = int(pred.ravel()[0])
    prediction_label = "HOTSPOT" if prediction == 1 else "NOT A HOTSPOT"

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "probability": round(prob_pos, 6),
    }


if __name__ == "__main__":
    # Example usage: a realistic sample input (may or may not be predicted as HOTSPOT)
    example = {
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

    print("Prediction pipeline test")
    print("------------------------")
    try:
        result = predict_hotspot(example)
    except Exception as exc:  # pragma: no cover - runtime interaction only
        print(f"Error running prediction example: {exc}", file=sys.stderr)
        raise

    label = result["prediction_label"]
    prob = result["probability"]
    print(f"Prediction: {label}")
    print(f"Probability: {prob * 100:.2f}%")
