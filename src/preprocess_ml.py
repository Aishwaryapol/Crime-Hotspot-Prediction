"""Preprocess ML inputs for Crime Hotspot Prediction.

This script loads train/test feature and target CSVs, validates feature
columns, builds a preprocessing ColumnTransformer (StandardScaler for
numerical, OneHotEncoder for categorical), fits it on `X_train` only,
transforms train/test, then saves the fitted preprocessor and feature
lists to `models/`.

DO NOT train any model here. This script is safe to run manually.
"""

from pathlib import Path
import json
import joblib

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    # Project root is parent of `src`
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    x_train_path = data_dir / "X_train.csv"
    x_test_path = data_dir / "X_test.csv"
    y_train_path = data_dir / "y_train.csv"
    y_test_path = data_dir / "y_test.csv"

    # Load datasets
    print(f"Loading: {x_train_path}")
    print(f"Loading: {x_test_path}")
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    print(f"Loading targets: {y_train_path}, {y_test_path}")
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    # --- Define features exactly as requested ---
    numerical_features = [
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

    categorical_features = [
        "DAY_OF_WEEK",
        "AREA NAME",
        "DOMINANT_CRIME_TYPE",
        "DOMINANT_PREMISE",
    ]

    approved_features = set(numerical_features + categorical_features)

    # --- Validation: CRIME_COUNT must not be present ---
    for name, df in [("X_train", X_train), ("X_test", X_test)]:
        if "CRIME_COUNT" in df.columns:
            raise ValueError(f"CRIME_COUNT found in {name} - this causes target leakage. Remove it.")

    # --- Validation: missing values detection ---
    missing_train = X_train.isna().sum()
    missing_train = missing_train[missing_train > 0]
    if not missing_train.empty:
        print("Missing values detected in X_train by column:")
        for col, cnt in missing_train.items():
            print(f" - {col}: {int(cnt)} missing")
        raise ValueError("X_train contains missing values. Aborting preprocessing.")

    missing_test = X_test.isna().sum()
    missing_test = missing_test[missing_test > 0]
    if not missing_test.empty:
        print("Missing values detected in X_test by column:")
        for col, cnt in missing_test.items():
            print(f" - {col}: {int(cnt)} missing")
        raise ValueError("X_test contains missing values. Aborting preprocessing.")

    # --- Validation: required columns exist ---
    missing_cols = {"X_train": [], "X_test": []}
    for col in approved_features:
        if col not in X_train.columns:
            missing_cols["X_train"].append(col)
        if col not in X_test.columns:
            missing_cols["X_test"].append(col)

    if missing_cols["X_train"] or missing_cols["X_test"]:
        if missing_cols["X_train"]:
            print("Missing in X_train:", sorted(missing_cols["X_train"]))
        if missing_cols["X_test"]:
            print("Missing in X_test:", sorted(missing_cols["X_test"]))
        raise ValueError("One or more required feature columns are missing from X_train/X_test.")

    # --- Validation: unexpected columns ---
    unexpected_train = [c for c in X_train.columns if c not in approved_features]
    unexpected_test = [c for c in X_test.columns if c not in approved_features]
    unexpected = sorted(set(unexpected_train + unexpected_test))
    if unexpected:
        print("Unexpected feature columns detected:")
        for c in unexpected:
            print(" -", c)
        raise ValueError("Unexpected feature columns present. Remove or whitelist them.")

    # --- Additional validation: numeric dtype check for numerical features ---
    non_numeric = []
    for col in numerical_features:
        if not is_numeric_dtype(X_train[col]):
            non_numeric.append(("X_train", col, X_train[col].dtype))
        if not is_numeric_dtype(X_test[col]):
            non_numeric.append(("X_test", col, X_test[col].dtype))
    if non_numeric:
        print("Non-numeric columns found among declared numerical features:")
        for dfname, col, dtype in non_numeric:
            print(f" - {dfname}: {col} (dtype={dtype})")
        raise ValueError("One or more declared numerical features are not numeric in the data. Adjust feature lists or convert types.")

    # --- Print dataset info before preprocessing ---
    print("Original X_train shape:", X_train.shape)
    print("Original X_test shape:", X_test.shape)
    print("Number of numerical features:", len(numerical_features))
    print("Number of categorical features:", len(categorical_features))

    # --- Build ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    # --- Fit ONLY on X_train (prevent leakage) ---
    print("Fitting preprocessor on X_train only...")
    preprocessor.fit(X_train)

    # --- Transform train and test ---
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # --- Print transformed shapes and target distributions ---
    print("Transformed X_train shape:", getattr(X_train_transformed, "shape", None))
    print("Transformed X_test shape:", getattr(X_test_transformed, "shape", None))

    print("Target training distribution:\n", y_train.value_counts().to_string())
    print("Target testing distribution:\n", y_test.value_counts().to_string())

    print("CRIME_COUNT present in X_train?", "CRIME_COUNT" in X_train.columns)
    print("Preprocessor fitted on X_train only: True (fit called before transforming X_test)")

    # --- Save preprocessor and feature lists ---
    preprocessor_out = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_out)
    print(f"Saved fitted preprocessor to: {preprocessor_out}")

    features_out = models_dir / "feature_columns.json"
    with features_out.open("w", encoding="utf-8") as fh:
        json.dump({
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
        }, fh, indent=2)
    print(f"Saved feature lists to: {features_out}")

    print("Preprocessing complete. No model was trained.")


if __name__ == "__main__":
    main()
