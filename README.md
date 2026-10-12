# Electricity Bill & Power Usage Predictor

## Project Overview
The Electricity Bill & Power Usage Predictor is a Machine Learning project that analyzes household electricity consumption. It predicts future energy usage, estimates electricity bills, detects unusual consumption patterns, and visualizes electricity usage trends.

The project follows an end-to-end Machine Learning workflow, including data preprocessing, feature engineering, model training, evaluation, prediction, and result visualization.

## Objectives
- Predict next-hour electricity consumption.
- Estimate monthly electricity bills.
- Detect unusual electricity usage using anomaly detection.
- Analyze hourly consumption patterns.
- Compare Machine Learning models using evaluation metrics.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
- **Model Saving:** Joblib
- **Tools:** VS Code, Git, GitHub

## Dataset
The project uses household electricity consumption data.

- **Original Records:** 2,075,259
- **Usable Hourly Records:** 33,791
- **Target Variable:** `Energy_kWh`

The dataset includes electricity measurements such as active power, voltage, current intensity, and sub-metering values. Data preprocessing and feature engineering are performed before model training.

## Machine Learning Models

**1. Linear Regression**  
Used as a baseline model for predicting electricity consumption.

**2. Random Forest Regressor**  
Used to capture more complex patterns in electricity usage.

**3. Isolation Forest**  
Used to identify unusual electricity consumption records.

## Model Performance

| Evaluation Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 0.3802 | **0.3437** |
| RMSE | 0.5314 | **0.4946** |
| R² Score | 0.4912 | **0.5593** |
| Cross-Validation MAE | 0.4309 | **0.4004** |

Random Forest performed better than Linear Regression across the reported evaluation metrics.

## Key Results
- **Predicted Next-Hour Consumption:** 1.554 kWh
- **Estimated Cost:** ₹12.43
- **Unusual Records Detected:** 676
- **Peak Usage Hour:** 8 PM
- **Lowest Average Usage Hour:** 4 AM
- **Tariff Assumption:** ₹8 per kWh

*Bill estimates use a flat tariff and do not include slab rates, fixed charges, taxes, or other adjustments.*

## Visualizations
The project generates the following graphs:
- Hourly electricity consumption
- Average electricity usage by hour
- Actual vs. predicted consumption
- Anomaly detection

## Project Outputs
The `outputs/` folder contains:
- `Power_Usage_Model.pkl`
- `Anomaly_Detection_Model.pkl`
- `model_comparison.csv`
- `test_predictions.csv`
- `next_hour_prediction.csv`
- `anomaly_detection_results.csv`
- `monthly_bill_estimates.csv`

The `images/` folder stores the generated visualizations.

## How to Run the Project

**1. Install the required libraries**
```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

**2. Place the dataset**  
Keep `household_power_consumption.txt` inside the `Data/` folder.

**3. Run the Python script**
```bash
python Electricity_Bill_Power_Usage_Predictor.py

## Future Improvements

- **Advanced Models:** Experiment with XGBoost and other advanced regression algorithms to improve prediction performance.
- **Hyperparameter Tuning:** Optimize model parameters to achieve better accuracy.
- **Real-Time Prediction:** Develop a system to predict electricity consumption using live meter data.
- **Accurate Bill Estimation:** Include electricity tariff slabs, fixed charges, and taxes for more realistic bill calculations.
- **Improved Anomaly Detection:** Explore advanced techniques to identify unusual consumption patterns.
- **Web Application:** Build an interactive dashboard using Streamlit to display predictions, estimated bills, and usage trends.
```

## Conclusion
This project demonstrates how Machine Learning can be applied to household energy consumption forecasting, bill estimation, anomaly detection, and usage analysis. Random Forest achieved an MAE of **0.3437** and an R² score of **0.5593** on the reported test set.

*Results are based on the current dataset and evaluation setup. Actual electricity consumption and bills may vary.*