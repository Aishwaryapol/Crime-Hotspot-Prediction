# Crime Hotspot Prediction
The project uses a Los Angeles crime dataset containing approximately 1 million crime records. Due to the large size of the processed datasets, the datasets are not included in this GitHub repository. The repository contains the complete preprocessing and feature-engineering scripts required to reproduce the datasets when the source data is available.
## Project Information

- Project Title: Crime Hotspot Prediction
- Student: Aishwarya Pol
- Institute: Sinhgad Institute, Vadgaon, Pune
- Project/Class Organization: Nirmaan Organization, Pune
- Mentor: Dhanashree Mam
- GitHub: The project has not been uploaded to GitHub yet.

This project was developed as a class/project under Nirmaan Organization, Pune, while the student is affiliated with Sinhgad Institute, Vadgaon, Pune.

## Abstract

This project investigates crime hotspot identification by combining historical crime data, geographic information, time-related information, dominant crime type, dominant premise, and historical lag features. The workflow aggregates crime activity at the weekly geographic grid-cell level, creates hotspot labels using a defined threshold, prepares a machine-learning dataset, trains and evaluates baseline classification models, and deploys an interactive Streamlit interface for prediction. The final system predicts the likelihood that a geographic grid-cell observation will be classified as a hotspot according to the project’s hotspot definition.

## Problem Statement

Crime incidents are unevenly distributed across urban areas and time periods. Identifying locations that are likely to become crime hotspots can assist with resource planning, situational awareness, and academic analysis. This project addresses that problem by building a machine-learning-based hotspot prediction workflow using historical crime data and a user-friendly predictive interface.

## Objectives

- Build a machine-learning workflow for crime hotspot prediction.
- Aggregate crime data into weekly geographic grid-cell observations.
- Generate hotspot labels based on a defined threshold.
- Develop historical lag features that avoid using current-week target leakage.
- Train and compare baseline classification models.
- Deploy an interactive Streamlit dashboard for prediction and analysis.
- Save prediction records for later review and analysis.

## Dataset

The project uses a Los Angeles crime dataset covering approximately 2020-01-01 through 2024-12-30.

Dataset source: Los Angeles crime dataset. The exact source URL should be added here before publishing the project to GitHub.

Important dataset facts established during development:

- Original cleaned dataset: 1,002,636 records
- Cleaned columns: 13
- Rows removed during cleaning: 2,240
- Date range: 2020-01-01 to 2024-12-30
- Latitude range: 33.7059 to 34.3343
- Longitude range: -118.6676 to -118.1554
- Unique AREA NAME values: 21
- Unique crime types: 140

## Data Preprocessing

The preprocessing workflow started from the cleaned crime dataset and focused on a selected set of key columns:

- DATE OCC
- YEAR
- MONTH
- DAY
- DAY_OF_WEEK
- HOUR
- AREA
- AREA NAME
- Crm Cd
- Crm Cd Desc
- Premis Desc
- LAT
- LON

The preprocessing steps included:

- Validation of missing values in the selected columns.
- Date parsing for DATE OCC values.
- Removal of rows that did not meet the defined cleaning criteria.
- Preparation of structured features for hotspot generation and machine-learning modeling.

Missing-value validation showed that no remaining missing values were present in the selected cleaned columns after preprocessing.

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the distribution of crime activity across time and geography.

Key findings include:

- Most common crime type: VEHICLE - STOLEN
- Other common categories included: BATTERY - SIMPLE ASSAULT, BURGLARY FROM VEHICLE, THEFT OF IDENTITY, VANDALISM - FELONY, BURGLARY, THEFT PLAIN - PETTY, ASSAULT WITH DEADLY WEAPON / AGGRAVATED ASSAULT, INTIMATE PARTNER - SIMPLE ASSAULT, and THEFT FROM MOTOR VEHICLE - PETTY
- Highest crime day: Friday
- Highest crime hour: 12
- Highest crime year: 2022
- Unique areas: 21
- Unique geographic grid cells: 1,279

## Hotspot Generation

The project generated weekly geographic grid-cell hotspot labels by aggregating crime counts at the weekly location level.

Aggregation level:

- WEEK + LAT_GRID + LON_GRID

Hotspot threshold:

- 75th percentile
- Threshold value: 6.0

The hotspot generation process produced:

- Initial weekly grid-cell records: 210,833
- Hotspot records: 58,267
- Non-hotspot records: 152,566
- Hotspot percentage: 27.64%
- Unique geographic grid cells: 1,279
- Unique weeks: 262

A record was labeled as:

- Hotspot = 1 when the weekly grid-cell crime count met the defined hotspot threshold.
- Hotspot = 0 otherwise.

This setup was later used in the machine-learning dataset, where current-week crime-count and hotspot labels were excluded from model features to prevent leakage.

## ML Dataset Preparation

The final machine-learning dataset was prepared from the processed hotspot dataset stored in data/processed/crime_hotspot.csv.

Key steps included:

- Weekly grid-cell deduplication.
- Creation of lag-based historical features.
- Removal of records lacking sufficient historical context.

Dataset statistics:

- Rows after weekly grid-cell deduplication: 210,833
- Rows after creating historical lag features and removing insufficient-history rows: 207,033
- Rows removed during this step: 3,800
- Final training samples: 171,992
- Final testing samples: 35,041
- Training period: 2020–2023
- Testing period: 2024

Training target distribution:

- Hotspot = 0: 120,622
- Hotspot = 1: 51,370

Testing target distribution:

- Hotspot = 0: 28,851
- Hotspot = 1: 6,190

## Feature Engineering

The final model features were designed to capture recent historical crime activity while avoiding target leakage from the current week.

Numerical features:

- YEAR
- MONTH
- HOUR
- AREA
- LAT_GRID
- LON_GRID
- PREV_WEEK_CRIME_COUNT
- PREV_2_WEEK_CRIME_COUNT
- PREV_3_WEEK_CRIME_COUNT
- ROLLING_4_WEEK_CRIME_COUNT
- GRID_TOTAL_PREVIOUS_CRIMES

Categorical features:

- DAY_OF_WEEK
- AREA NAME
- DOMINANT_CRIME_TYPE
- DOMINANT_PREMISE

Historical features:

- PREV_WEEK_CRIME_COUNT: crime count from the previous week for the same grid cell.
- PREV_2_WEEK_CRIME_COUNT: crime count from two weeks earlier.
- PREV_3_WEEK_CRIME_COUNT: crime count from three weeks earlier.
- ROLLING_4_WEEK_CRIME_COUNT: rolling crime count over the previous four weeks, excluding the current week.
- GRID_TOTAL_PREVIOUS_CRIMES: cumulative historical crime count for the grid cell, excluding the current week.

Current-week CRIME_COUNT and Hotspot were excluded from model features to prevent leakage and preserve the intended predictive setup.

## Preprocessing Pipeline

The preprocessing pipeline used a ColumnTransformer with:

- StandardScaler for numerical features
- OneHotEncoder(handle_unknown="ignore") for categorical features

The preprocessor was fitted only on the training set (X_train) and then used to transform the test set. This approach is important because it helps prevent data leakage from the test set into preprocessing statistics.

Key preprocessing details:

- Transformed feature count: 448
- Training transformed shape: 171,992 × 448
- Testing transformed shape: 35,041 × 448
- Saved artifact: models/preprocessor.joblib
- Feature metadata: models/feature_columns.json

## Machine Learning Models

The project evaluated three baseline models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Random Forest was selected as the final model because it provided the strongest overall baseline performance among the evaluated models.

## Model Performance

### Logistic Regression

Test results:

- Accuracy: 0.77
- Class 0 Precision: 0.97
- Class 0 Recall: 0.75
- Class 0 F1: 0.85
- Class 1 Precision: 0.43
- Class 1 Recall: 0.89
- Class 1 F1: 0.58

Confusion matrix:

- Actual 0 / Predicted 0: 21,656
- Actual 0 / Predicted 1: 7,195
- Actual 1 / Predicted 0: 695
- Actual 1 / Predicted 1: 5,495

Parameters:

- class_weight="balanced"
- max_iter=1000
- random_state=42

### Decision Tree

Test results:

- Accuracy: 0.71
- Class 0 Precision: 0.97
- Class 0 Recall: 0.67
- Class 0 F1: 0.79
- Class 1 Precision: 0.37
- Class 1 Recall: 0.91
- Class 1 F1: 0.53

Confusion matrix:

- Actual 0 / Predicted 0: 19,306
- Actual 0 / Predicted 1: 9,545
- Actual 1 / Predicted 0: 536
- Actual 1 / Predicted 1: 5,654

Parameters:

- random_state=42
- class_weight="balanced"
- max_depth=10
- min_samples_leaf=10

### Random Forest

The selected final model is a Random Forest Classifier with the following parameters:

- n_estimators=150
- max_depth=15
- min_samples_leaf=5
- class_weight="balanced"
- random_state=42
- n_jobs=-1

Test results:

- Accuracy: 0.82
- Class 0 Precision: 0.96
- Class 0 Recall: 0.82
- Class 0 F1: 0.88
- Class 1 Precision: 0.50
- Class 1 Recall: 0.86
- Class 1 F1: 0.63

Confusion matrix:

- Actual 0 / Predicted 0: 23,528
- Actual 0 / Predicted 1: 5,323
- Actual 1 / Predicted 0: 880
- Actual 1 / Predicted 1: 5,310

## Random Forest Feature Importance

Feature importance analysis was performed for the trained Random Forest model.

Total transformed features: 448

Top important transformed features:

1. ROLLING_4_WEEK_CRIME_COUNT — 0.230409
2. PREV_3_WEEK_CRIME_COUNT — 0.162559
3. PREV_2_WEEK_CRIME_COUNT — 0.139766
4. PREV_WEEK_CRIME_COUNT — 0.130442
5. GRID_TOTAL_PREVIOUS_CRIMES — 0.119879
6. LAT_GRID — 0.040266
7. LON_GRID — 0.035262
8. AREA — 0.017297
9. AREA NAME - Devonshire — 0.012519
10. DOMINANT_PREMISE - STREET — 0.010057

These results indicate that historical crime activity is the dominant source of predictive importance in the trained Random Forest model.

Saved files:

- models/random_forest_feature_importance.csv
- models/top_20_feature_importance.csv
- visualizations/random_forest_top_20_features.png

## Prediction Pipeline

The prediction pipeline is implemented in src/predict.py.

The workflow is:

1. Load the trained Random Forest model.
2. Load the fitted preprocessing pipeline.
3. Receive user input from the Streamlit interface.
4. Validate the feature names.
5. Prevent CRIME_COUNT and Hotspot from being supplied as input.
6. Apply the existing preprocessor using transform().
7. Generate the prediction.
8. Generate the probability for the positive class.
9. Return HOTSPOT or NOT A HOTSPOT.

The prediction pipeline does not retrain the model, fit the preprocessor, or modify datasets.

Example successful test:

- Prediction: HOTSPOT
- Probability: 83.39%

## Interactive Streamlit Dashboard

The Streamlit application provides an interactive dashboard for prediction and analysis.

Features include:

- Responsive design for desktop, laptop, tablet, and mobile screen sizes
- Prediction form with date, time, geographic, crime, and historical-input sections
- Prediction result card with hotspot probability
- Prediction history display
- Dashboard statistics and charts
- Geographic map display where valid coordinates are available
- CSV download for prediction history
- Model information and project statistics

The UI uses the existing prediction and CSV-saving functions and does not train models during runtime.

## Prediction History CSV

Every successful user prediction is saved to the project prediction history file:

- data/predictions/prediction_history.csv

The application appends new prediction records rather than deleting previous ones. The CSV contains the following important columns:

- timestamp
- YEAR
- MONTH
- DAY_OF_WEEK
- HOUR
- AREA
- AREA_NAME
- LAT_GRID
- LON_GRID
- DOMINANT_CRIME_TYPE
- DOMINANT_PREMISE
- PREV_WEEK_CRIME_COUNT
- PREV_2_WEEK_CRIME_COUNT
- PREV_3_WEEK_CRIME_COUNT
- ROLLING_4_WEEK_CRIME_COUNT
- GRID_TOTAL_PREVIOUS_CRIMES
- prediction
- prediction_label
- probability

The CSV can be downloaded from the Streamlit interface and later used for analysis.

## Project Structure

The current project structure is as follows:

```text
Crime-Hotspot-Prediction/
│
├── app.py
├── app_backup.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── crime_hotspot.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   ├── ml_dataset.csv
│   │   └── feature_info.txt
│   │
│   └── predictions/
│       └── prediction_history.csv
│
├── models/
│   ├── preprocessor.joblib
│   ├── feature_columns.json
│   ├── logistic_regression.joblib
│   ├── logistic_regression_metrics.json
│   ├── decision_tree.joblib
│   ├── decision_tree_metrics.json
│   ├── random_forest.joblib
│   ├── random_forest_metrics.json
│   ├── random_forest_feature_importance.csv
│   └── top_20_feature_importance.csv
│
├── notebooks/
├── src/
│   ├── inspect_data.py
│   ├── create_ml_features.py
│   ├── preprocess_ml.py
│   ├── train_logistic_regression.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── analyze_feature_importance.py
│   ├── predict.py
│   └── save_prediction.py
│
└── visualizations/
    └── random_forest_top_20_features.png
```

## Installation

Follow these steps to set up the project locally.

1. Clone or download the project folder.
2. Open the project directory.
3. Create a virtual environment.
4. Activate the virtual environment.
5. Install the dependencies.
6. Run the Streamlit application.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How to Run

After installing the dependencies, run:

```bash
streamlit run app.py
```

The application will open in the browser and allow the user to submit prediction inputs, view the prediction result, and review the saved history.

## Example Prediction

A typical example prediction uses the following type of input:

- Year: 2024
- Month: December
- Day of week: Saturday
- Hour: 22
- Area: 1
- Area Name: Central
- Historical crime activity features as entered by the user

Example successful test output:

- Prediction: HOTSPOT
- Probability: 83.39%

## Screenshots

Screenshots should be added before final GitHub upload. Suggested placeholder locations are:

- docs/screenshots/dashboard.png
- docs/screenshots/prediction.png
- docs/screenshots/history.png
- docs/screenshots/feature-importance.png

## GitHub Upload

The project has not been uploaded to GitHub yet. To publish it later, the student can initialize a Git repository and push the project to a new repository.

Example commands:

```bash
git init
git add .
git commit -m "Initial Crime Hotspot Prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/Crime-Hotspot-Prediction
git push -u origin main
```

Before uploading to GitHub, avoid committing sensitive or local-only files. Recommended local-only items include:

- .venv/
- __pycache__/
- *.pyc
- .vscode/
- .env

The large raw dataset should be considered carefully before uploading to GitHub, depending on repository size limits and project-sharing requirements.

## Limitations

- The model depends on historical crime data and may not capture sudden behavioral changes.
- Prediction quality depends on the quality of the input features provided by the user.
- The model does not guarantee future crime occurrence.
- The dataset represents reported and recorded crime activity and may not reflect all incidents.
- The hotspot definition depends on the selected 75th percentile threshold.
- Geographic grid resolution affects the interpretation of results.
- Historical patterns can change over time.
- This is an academic machine-learning project and not a law-enforcement decision system.

## Future Scope

Possible future enhancements include:

- Real-time crime data integration
- More advanced spatial-temporal models
- Hyperparameter optimization
- Model monitoring and calibration
- More detailed geographic analysis
- Explainable AI methods
- Automated retraining workflows
- Cloud deployment
- Improved map visualization
- Additional external contextual features such as population or environmental data

## Conclusion

This project demonstrates a complete academic machine-learning workflow for crime hotspot prediction using historical crime data, engineered lag features, and a Random Forest-based classifier. The final Streamlit application provides a practical interface for making predictions, reviewing results, and saving prediction history for further analysis.

## Author

Aishwarya Pol

Sinhgad Institute, Vadgaon, Pune

Nirmaan Organization, Pune

