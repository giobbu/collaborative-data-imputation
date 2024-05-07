import mlflow
import numpy as np
from source.models.naive.mlflow_naive_imputation import SimpleAvgImputation


def run_grouped_avg_experiment(training_df, validation_df, **params):

    # Simple Avg Imputation
    simple_avg_imputation = SimpleAvgImputation(variable_name="power_z", group_by = params['group_by'])

    # Saving the model with mlflow
    with mlflow.start_run(run_name = params['group_by']):

        # Train Model
        simple_avg_imputation.fit(training_df)

        # Evaluate the model
        validation_predictions = simple_avg_imputation.predict(validation_df)
        eval_rmse = simple_avg_imputation.compute_rmse(np.array(validation_df.power_z), validation_predictions)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic Model for Data Imputation: Simple average grouped by periodId ")

        # Log params and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)

        # Save model
        mlflow.pyfunc.log_model(artifact_path = "simple_avg_model",
                                            python_model = simple_avg_imputation,
                                            registered_model_name= params['model'] +'-' + params['group_by'],
                                            )