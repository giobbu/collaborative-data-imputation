import numpy as np
from mlflow.pyfunc import PythonModel


class SimpleAvgImputation(PythonModel):
    def __init__(self, variable_name, group_by):
        self.variable_name = variable_name
        self.group_by = group_by
        self.avg_mean = None
    
    def fit(self, training_df):
        try:
            self.avg_mean = training_df.groupby(self.group_by)[self.variable_name].mean()
        except Exception as e:
            print(f"Error during fitting: {e}")

    def predict(self, test_df):
        try:
            predictions = self.avg_mean[test_df[self.group_by]].values
            return predictions
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    @staticmethod
    def compute_rmse(targets, predictions):
        try:
            rmse = np.sqrt(np.sum((predictions - targets) ** 2) / len(predictions))
            return rmse
        except Exception as e:
            print(f"Error computing RMSE: {e}")
            return None
