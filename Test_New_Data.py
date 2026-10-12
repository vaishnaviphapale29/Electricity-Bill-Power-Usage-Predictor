
import unittest
import pandas as pd
import numpy as np

from Electricity_Bill_Power_Usage_Predictor import (
    LoadDataset,
    PrepareData,
    TrainModels,
    DetectAnomalies,
    OUTPUT_DIR
)


class TestElectricityPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.RawData = LoadDataset()
        cls.Data = PrepareData(cls.RawData)

        # Keep tests fast.
        cls.TestData = cls.Data.tail(500).copy()

    def test_dataset_loaded(self):
        self.assertFalse(self.RawData.empty)

    def test_preprocessing(self):
        self.assertIn("Energy_kWh", self.Data.columns)
        self.assertFalse(self.Data.empty)

    def test_model_training(self):

        TrainModels(self.TestData)

        ModelPath = OUTPUT_DIR / "Power_Usage_Model.pkl"

        self.assertTrue(ModelPath.exists())

    def test_prediction(self):

        import joblib

        ModelPath = OUTPUT_DIR / "Power_Usage_Model.pkl"

        if not ModelPath.exists():
            TrainModels(self.TestData)

        ModelBundle = joblib.load(ModelPath)

        Model = ModelBundle["model"]
        Features = ModelBundle["feature_columns"]

        InputData = self.TestData[Features].tail(1)

        Prediction = Model.predict(InputData)

        self.assertEqual(len(Prediction), 1)
        self.assertTrue(np.isfinite(Prediction[0]))
        self.assertGreaterEqual(Prediction[0], 0)

    def test_anomaly_detection(self):

        DetectAnomalies(self.TestData)

        ResultPath = (
            OUTPUT_DIR / "anomaly_detection_results.csv"
        )

        self.assertTrue(ResultPath.exists())

        Results = pd.read_csv(ResultPath)

        self.assertIn("Anomaly", Results.columns)
        self.assertFalse(Results.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
