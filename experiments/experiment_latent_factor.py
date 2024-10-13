import mlflow
import numpy as np
from source.models.latent_factor.mlflow_latent_factor import LatentFactorModel


def run_latent_factor_experiment(training_df, validation_df, **params):
    " Run latent factor model experiment"

    # Latent Factor Model
    lf_model = LatentFactorModel(k=params['latent_dimensions'], var_name='power_z', warm_start=params['warm_start'], verbose=True, learning_rate=params['learning_rate'], random_state=params['seed'])

    with mlflow.start_run(run_name=params['solver']):

        # Train Model
        U , P = lf_model.factorization(train_df=training_df,
                                        valid_df=validation_df,
                                        solver=params['solver'],
                                        n_epochs=params['n_epochs'],
                                        lambda_reg_P=params['lambda_reg_P'],
                                        lambda_reg_U=params['lambda_reg_U'])

        # Evaluate the model
        validation_predictions = lf_model.predict(U, P, validation_df)
        eval_rmse = lf_model.compute_rmse(np.array(validation_df.power_z), validation_predictions)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Latent Factor Model for Matrix Factorization ")

        # Log params and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)

        # Save model
        mlflow.pyfunc.log_model(artifact_path = params['model'],
                                python_model = lf_model,
                                #registered_model_name= params['model'] +'-' + params['solver'],
                                )
