import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import joblib

from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# =====================================================
# PROJECT SETTINGS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Data" / "household_power_consumption.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR = BASE_DIR / "images"

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

TARIFF_PER_UNIT = 8.0

logging.basicConfig(
    filename=OUTPUT_DIR / "project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================================================
# STEP 1: LOAD DATASET
# =====================================================

def LoadDataset():

    print("\nSTEP 1: LOAD DATASET")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    Data = pd.read_csv(
        DATA_PATH,
        sep=";",
        na_values=["?"],
        low_memory=False
    )

    Data["DateTime"] = pd.to_datetime(
        Data["Date"] + " " + Data["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    NumericColumns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for Column in NumericColumns:
        Data[Column] = pd.to_numeric(
            Data[Column], errors="coerce"
        )

    Data = Data.dropna(subset=["DateTime"])
    Data = Data.sort_values("DateTime")
    Data = Data.drop_duplicates(subset=["DateTime"])

    print("Dataset loaded successfully.")
    print("Total rows:", len(Data))

    logging.info("Dataset loaded successfully")

    return Data

# =====================================================
# STEP 2: PREPROCESS DATA
# =====================================================

def PrepareData(Data):

    print("\nSTEP 2: DATA PREPROCESSING")

    Data = Data.set_index("DateTime")

    # Convert minute-wise power into hourly energy.
    HourlyEnergy = (
        Data["Global_active_power"]
        .resample("h")
        .sum(min_count=1) / 60
    )

    # Require at least 45 valid readings per hour.
    ValidReadings = (
        Data["Global_active_power"]
        .resample("h")
        .count()
    )

    HourlyEnergy[ValidReadings < 45] = np.nan

    HourlyData = pd.DataFrame({
        "Energy_kWh": HourlyEnergy
    })

    MeterColumns = [
        "Global_intensity",
        "Voltage",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for Column in MeterColumns:
        HourlyData[Column] = (
            Data[Column].resample("h").mean()
        )

    # Time features.
    HourlyData["Hour"] = HourlyData.index.hour
    HourlyData["Day"] = HourlyData.index.day
    HourlyData["Month"] = HourlyData.index.month
    HourlyData["DayOfWeek"] = HourlyData.index.dayofweek

    # Previous consumption features.
    HourlyData["Previous_Hour_Usage"] = (
        HourlyData["Energy_kWh"].shift(1)
    )

    HourlyData["Previous_Day_Usage"] = (
        HourlyData["Energy_kWh"].shift(24)
    )

    HourlyData["Rolling_24_Hour_Usage"] = (
        HourlyData["Energy_kWh"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    HourlyData = HourlyData.replace(
        [np.inf, -np.inf], np.nan
    )

    HourlyData = HourlyData.dropna(
        subset=[
            "Energy_kWh",
            "Previous_Hour_Usage",
            "Previous_Day_Usage",
            "Rolling_24_Hour_Usage"
        ]
    )

    print("Preprocessing completed.")
    print("Usable hourly records:", len(HourlyData))

    return HourlyData

# =====================================================
# STEP 3: CREATE AND DISPLAY GRAPHS
# =====================================================

def CreateGraphs(Data):

    print("\nSTEP 3: DATA VISUALIZATION")

    # Graph 1: Hourly consumption.
    plt.figure(figsize=(12, 5))

    plt.plot(
        Data.index,
        Data["Energy_kWh"],
        label="Hourly Usage"
    )

    plt.title("Hourly Electricity Consumption")
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "hourly_consumption.png")
    plt.show()
    plt.close()

    # Graph 2: Average usage by hour.
    HourlyAverage = (
        Data.groupby("Hour")["Energy_kWh"].mean()
    )

    plt.figure(figsize=(10, 5))

    HourlyAverage.plot(kind="bar")

    plt.title("Average Electricity Usage by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Energy (kWh)")
    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "usage_by_hour.png")
    plt.show()
    plt.close()

    print("Graphs saved in images folder.")

# =====================================================
# STEP 4: TRAIN AND EVALUATE MODELS
# =====================================================

def TrainModels(Data):

    print("\nSTEP 4: MODEL TRAINING")

    FeatureColumns = [
        "Hour",
        "Day",
        "Month",
        "DayOfWeek",
        "Previous_Hour_Usage",
        "Previous_Day_Usage",
        "Rolling_24_Hour_Usage"
    ]

    X = Data[FeatureColumns]
    Y = Data["Energy_kWh"]

    if len(Data) < 30:
        raise ValueError("Not enough hourly records.")

    # Keep chronological order.
    SplitIndex = int(len(Data) * 0.80)

    X_Train = X.iloc[:SplitIndex]
    X_Test = X.iloc[SplitIndex:]

    Y_Train = Y.iloc[:SplitIndex]
    Y_Test = Y.iloc[SplitIndex:]

    Models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression())
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    # Time-series cross-validation.
    print("\nTIME-SERIES CROSS-VALIDATION")

    NumberOfSplits = min(5, len(X_Train) // 10)

    if NumberOfSplits >= 2:

        TimeSplit = TimeSeriesSplit(
            n_splits=NumberOfSplits
        )

        for ModelName, Model in Models.items():

            Scores = cross_val_score(
                Model,
                X_Train,
                Y_Train,
                cv=TimeSplit,
                scoring="neg_mean_absolute_error",
                n_jobs=1
            )

            print(
                ModelName,
                "Average CV MAE:",
                round((-Scores).mean(), 4)
            )

    Results = []
    BestModel = None
    BestModelName = None
    BestMAE = float("inf")

    # Train and compare.
    for ModelName, Model in Models.items():

        print("\nTraining:", ModelName)

        Model.fit(X_Train, Y_Train)

        Predictions = Model.predict(X_Test)

        MAE = mean_absolute_error(Y_Test, Predictions)

        RMSE = np.sqrt(
            mean_squared_error(Y_Test, Predictions)
        )

        R2 = r2_score(Y_Test, Predictions)

        print("MAE:", round(MAE, 4))
        print("RMSE:", round(RMSE, 4))
        print("R2 Score:", round(R2, 4))

        Results.append({
            "Model": ModelName,
            "MAE": MAE,
            "RMSE": RMSE,
            "R2": R2
        })

        if MAE < BestMAE:
            BestMAE = MAE
            BestModel = Model
            BestModelName = ModelName

    pd.DataFrame(Results).to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # Save test predictions.
    BestPredictions = np.maximum(
        BestModel.predict(X_Test), 0
    )

    PredictionData = pd.DataFrame({
        "DateTime": Data.index[SplitIndex:],
        "Actual_kWh": Y_Test.to_numpy(),
        "Predicted_kWh": BestPredictions
    })

    PredictionData.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    # Graph 3: Actual vs predicted.
    GraphData = PredictionData.iloc[:200]

    plt.figure(figsize=(12, 5))

    plt.plot(
        GraphData["DateTime"],
        GraphData["Actual_kWh"],
        label="Actual"
    )

    plt.plot(
        GraphData["DateTime"],
        GraphData["Predicted_kWh"],
        label="Predicted"
    )

    plt.title("Actual vs Predicted Electricity Usage")
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "actual_vs_predicted.png")
    plt.show()
    plt.close()

    # Save model and feature names.
    joblib.dump(
        {
            "model": BestModel,
            "feature_columns": FeatureColumns,
            "model_name": BestModelName
        },
        OUTPUT_DIR / "Power_Usage_Model.pkl"
    )

    print("\nBest model:", BestModelName)
    print("Model saved successfully.")

    return BestModel

# =====================================================
# STEP 5: NEXT-HOUR PREDICTION
# =====================================================

def PredictNextHour(Data):

    print("\nSTEP 5: NEXT-HOUR PREDICTION")

    ModelPath = OUTPUT_DIR / "Power_Usage_Model.pkl"

    if not ModelPath.exists():
        print("Please train the model first.")
        return

    RecentData = Data.tail(24)

    if len(RecentData) < 24:
        print("Not enough hourly readings.")
        return

    TimeDifference = RecentData.index.to_series().diff()

    if not (
        TimeDifference.iloc[1:] == pd.Timedelta(hours=1)
    ).all():

        print("Recent readings contain missing hours.")
        print("Next-hour prediction skipped.")
        return

    ModelBundle = joblib.load(ModelPath)

    Model = ModelBundle["model"]
    FeatureColumns = ModelBundle["feature_columns"]

    LastTime = Data.index[-1]
    NextTime = LastTime + pd.Timedelta(hours=1)

    PreviousHourUsage = Data["Energy_kWh"].iloc[-1]
    PreviousDayUsage = Data["Energy_kWh"].iloc[-24]

    RollingUsage = RecentData["Energy_kWh"].mean()

    InputData = pd.DataFrame([{
        "Hour": NextTime.hour,
        "Day": NextTime.day,
        "Month": NextTime.month,
        "DayOfWeek": NextTime.dayofweek,
        "Previous_Hour_Usage": PreviousHourUsage,
        "Previous_Day_Usage": PreviousDayUsage,
        "Rolling_24_Hour_Usage": RollingUsage
    }])

    Prediction = Model.predict(
        InputData[FeatureColumns]
    )[0]

    Prediction = max(0, Prediction)

    EstimatedCost = Prediction * TARIFF_PER_UNIT

    print("Prediction time:", NextTime)
    print(
        "Predicted energy:",
        round(Prediction, 3), "kWh"
    )
    print(
        "Estimated cost: Rs.",
        round(EstimatedCost, 2)
    )

    pd.DataFrame([{
        "Prediction_Time": NextTime,
        "Predicted_Energy_kWh": Prediction,
        "Estimated_Cost_Rs": EstimatedCost
    }]).to_csv(
        OUTPUT_DIR / "next_hour_prediction.csv",
        index=False
    )

# =====================================================
# STEP 6: ANOMALY DETECTION
# =====================================================

def DetectAnomalies(Data):

    print("\nSTEP 6: ANOMALY DETECTION")

    AnomalyColumns = [
        "Energy_kWh",
        "Global_intensity",
        "Voltage",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    AnomalyData = Data[AnomalyColumns].copy()

    Imputer = SimpleImputer(strategy="median")
    CleanData = Imputer.fit_transform(AnomalyData)

    Model = IsolationForest(
        contamination=0.02,
        random_state=42
    )

    Labels = Model.fit_predict(CleanData)

    Results = Data.copy()

    Results["Anomaly"] = np.where(
        Labels == -1,
        "Unusual",
        "Normal"
    )

    Results.to_csv(
        OUTPUT_DIR / "anomaly_detection_results.csv"
    )

    joblib.dump(
        {
            "model": Model,
            "imputer": Imputer,
            "feature_columns": AnomalyColumns
        },
        OUTPUT_DIR / "Anomaly_Detection_Model.pkl"
    )

    print(
        "Unusual records detected:",
        int(np.sum(Labels == -1))
    )

    NormalData = Results[Results["Anomaly"] == "Normal"]
    UnusualData = Results[Results["Anomaly"] == "Unusual"]

    # Graph 4: Anomaly detection.
    plt.figure(figsize=(10, 5))

    plt.scatter(
        NormalData["Hour"],
        NormalData["Energy_kWh"],
        label="Normal",
        alpha=0.5
    )

    plt.scatter(
        UnusualData["Hour"],
        UnusualData["Energy_kWh"],
        label="Unusual",
        alpha=0.8
    )

    plt.title("Unusual Electricity Usage")
    plt.xlabel("Hour of Day")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "anomaly_detection.png")
    plt.show()
    plt.close()

    print("Anomaly results saved.")

# =====================================================
# STEP 7: MONTHLY BILL ESTIMATION
# =====================================================

def EstimateBill(Data):

    print("\nSTEP 7: MONTHLY BILL ESTIMATION")

    MonthlyUsage = (
        Data["Energy_kWh"]
        .resample("MS")
        .sum(min_count=1)
    )

    BillData = MonthlyUsage.to_frame(
        name="Estimated_Units_kWh"
    )

    BillData["Estimated_Bill_Rs"] = (
        BillData["Estimated_Units_kWh"]
        * TARIFF_PER_UNIT
    )

    BillData.to_csv(
        OUTPUT_DIR / "monthly_bill_estimates.csv"
    )

    print(BillData.tail())

    print(
        "\nExample tariff: Rs.",
        TARIFF_PER_UNIT,
        "per kWh"
    )

    print(
        "This is an estimate using a flat rate. "
        "It excludes slab rates, fixed charges, taxes "
        "and other bill adjustments."
    )

# =====================================================
# STEP 8: ENERGY USAGE INSIGHTS
# =====================================================

def ShowInsights(Data):

    print("\nSTEP 8: ENERGY SAVING INSIGHTS")

    AverageUsage = (
        Data.groupby("Hour")["Energy_kWh"].mean()
    )

    PeakHour = AverageUsage.idxmax()
    LowUsageHour = AverageUsage.idxmin()

    print("Peak usage hour:", PeakHour)
    print("Lowest average usage hour:", LowUsageHour)

    print(
        "Peak average consumption:",
        round(AverageUsage.max(), 3), "kWh"
    )

    print(
        "Lowest average consumption:",
        round(AverageUsage.min(), 3), "kWh"
    )

    print(
        "These are historical patterns, not proof "
        "of a specific appliance wasting electricity."
    )

# =====================================================
# MAIN FUNCTION
# =====================================================

def main():

    try:

        print("=" * 55)
        print("ELECTRICITY BILL & POWER USAGE PREDICTOR")
        print("=" * 55)

        Data = LoadDataset()

        HourlyData = PrepareData(Data)

        if len(HourlyData) < 30:
            raise ValueError(
                "Not enough usable hourly records."
            )

        CreateGraphs(HourlyData)

        TrainModels(HourlyData)

        PredictNextHour(HourlyData)

        DetectAnomalies(HourlyData)

        EstimateBill(HourlyData)

        ShowInsights(HourlyData)

        print("\nPROJECT COMPLETED SUCCESSFULLY!")
        print("Check outputs and images folders.")

        logging.info("Project completed successfully")

    except Exception as Error:

        logging.exception("Project failed")
        print("\nERROR:", Error)

if __name__ == "__main__":
    main()