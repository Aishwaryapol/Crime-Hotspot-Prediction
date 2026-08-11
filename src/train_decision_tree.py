"""Train and evaluate a Decision Tree baseline for Crime Hotspot Prediction.

This script trains a DecisionTreeClassifier on preprocessed features.
It loads the pre-fitted preprocessing transformer from `models/preprocessor.joblib`.

Do NOT fit the preprocessor here. Do NOT use the test set for training.

Run manually:
    python src/train_decision_tree.py
"""

from pathlib import Path
import json
import joblib

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def main():
    # Resolve project root and data/model paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Input file paths
    x_train_path = data_dir / "X_train.csv"
    x_test_path = data_dir / "X_test.csv"
    y_train_path = data_dir / "y_train.csv"
    y_test_path = data_dir / "y_test.csv"
    preprocessor_path = models_dir / "preprocessor.joblib"

    # Load features and targets
    print(f"Loading features: {x_train_path} and {x_test_path}")
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    print(f"Loading targets: {y_train_path} and {y_test_path}")
    y_train_df = pd.read_csv(y_train_path)
    y_test_df = pd.read_csv(y_test_path)

    # --- CRIME_COUNT leakage check ---
    if "CRIME_COUNT" in X_train.columns or "CRIME_COUNT" in X_test.columns:
        raise ValueError("CRIME_COUNT found in features — this causes target leakage. Remove it before training.")
    print("CRIME_COUNT leakage check: PASSED")

    # --- Extract target series (Hotspot) ---
    if "Hotspot" in y_train_df.columns:
        y_train = y_train_df["Hotspot"].copy()
    else:
        y_train = y_train_df.iloc[:, 0].copy()

    if "Hotspot" in y_test_df.columns:
        y_test = y_test_df["Hotspot"].copy()
    else:
        y_test = y_test_df.iloc[:, 0].copy()

    # Print sample counts
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # --- Load pre-fitted preprocessor (do not fit here) ---
    print(f"Loading preprocessor from: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    # --- Transform features ---
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Print transformed shapes
    print("Training features after preprocessing:", getattr(X_train_transformed, "shape", None))
    print("Testing features after preprocessing:", getattr(X_test_transformed, "shape", None))

    # --- Initialize Decision Tree with baseline parameters ---
    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
        max_depth=10,
        min_samples_leaf=10,
    )

    # --- Train on training data only ---
    print("Training Decision Tree...")
    clf.fit(X_train_transformed, y_train)

    # --- Predictions and probabilities on test set ---
    y_pred = clf.predict(X_test_transformed)
    # predict_proba returns two columns for binary classification; take positive class
    y_prob = clf.predict_proba(X_test_transformed)[:, 1]

    # --- Evaluation metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = None

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix (readable)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    # Save trained model
    model_out = models_dir / "decision_tree.joblib"
    joblib.dump(clf, model_out)
    print(f"Saved Decision Tree to: {model_out}")

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": (float(roc_auc) if roc_auc is not None else None),
    }
    metrics_out = models_dir / "decision_tree_metrics.json"
    with metrics_out.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics to: {metrics_out}")

    print("Done. No other models were trained.")


if __name__ == "__main__":
    main()
