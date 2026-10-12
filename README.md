# Electricity Bill & Power Usage Predictor

**An End-to-End Machine Learning Project for Electricity Consumption Forecasting, Anomaly Detection, and Bill Estimation**

## Overview

The Electricity Bill & Power Usage Predictor is a machine learning project designed to analyze historical household electricity consumption, forecast hourly power usage, identify abnormal consumption patterns, and estimate electricity costs.

The project demonstrates an end-to-end workflow covering data preprocessing, feature engineering, model training, evaluation, prediction, visualization, and automated testing.

## Features

- Hourly electricity consumption forecasting
- Data cleaning and preprocessing
- Time-based and historical feature engineering
- Linear Regression and Random Forest model comparison
- Model evaluation and time-series cross-validation
- Electricity consumption visualization
- Anomaly detection using Isolation Forest
- Estimated electricity bill calculation
- Trained model persistence using Joblib
- Automated unit testing using Unittest

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Testing:** Unittest

## Dataset

**Individual Household Electric Power Consumption**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains historical household electricity consumption measurements used to prepare hourly usage records and train machine learning models.

## Project Structure

```text
Electricity_Bill_Power_Usage_Predictor/
├── Data/
│   └── household_power_consumption.txt
├── images/
├── outputs/
├── Electricity_Bill_Power_Usage_Predictor.py
├── Test_New_Data.py
├── requirements.txt
└── README.md
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset is available at the configured path inside the `Data/` directory.

## Usage

**Run the main application:**

```bash
python Electricity_Bill_Power_Usage_Predictor.py
```

**Run automated tests:**

```bash
python -m unittest Test_New_Data.py -v
```

Run the main application first when testing functionality that requires a saved model.

## Machine Learning Workflow

1. Load the electricity consumption dataset.
2. Clean missing and invalid values.
3. Prepare hourly consumption records.
4. Engineer time-based and historical usage features.
5. Split data chronologically into training and testing sets.
6. Train and compare machine learning models.
7. Evaluate predictive performance.
8. Generate electricity usage predictions and visualizations.
9. Detect unusual consumption patterns.
10. Estimate electricity costs and save generated results.

## Expected Outputs

Depending on the configuration and successful execution, the project can generate:

- Electricity consumption visualizations
- Model comparison and evaluation results
- Actual versus predicted consumption
- Saved trained models
- Anomaly detection results
- Hourly usage predictions
- Estimated electricity bills

## Future Enhancements

- Explore advanced regression models such as XGBoost.
- Implement LSTM networks for time-series forecasting.
- Develop an interactive dashboard using Streamlit.
- Integrate weather data to analyze consumption patterns.
- Support dynamic electricity tariffs and billing slabs.
- Generate personalized energy-saving recommendations.
- Deploy the model through a web application or REST API.
- Implement automated model retraining with new data.

## Limitations

- Prediction performance depends on data quality and model validation.
- Future electricity consumption may differ from historical patterns.
- Bill estimates depend on the configured tariff and may not include all utility charges.