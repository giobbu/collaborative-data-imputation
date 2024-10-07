import numpy as np
from mlflow.pyfunc import PythonModel


class SimpleAvgImputation(PythonModel):
    " Simple average imputation model. "

    def __init__(self, variable_name, group_by):

        assert isinstance(variable_name, str), "Input variable_name must be a string."
        assert isinstance(group_by, str), "Input group_by must be a string."
        
        self.variable_name = variable_name
        self.group_by = group_by
        self.avg_mean = None
    
    def fit(self, training_df):
        " Fit the model. "
        self.avg_mean = training_df.groupby(self.group_by)[self.variable_name].mean()

    def predict(self, test_df):
        " Predict using the model. "
        predictions = self.avg_mean[test_df[self.group_by]].values
        return predictions

    @staticmethod
    def compute_rmse(targets, predictions):
        " Compute the root mean squared error. "
        rmse = np.sqrt(np.sum((predictions - targets) ** 2) / len(predictions))
        return rmse

