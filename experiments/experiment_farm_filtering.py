import mlflow
from source.utils.utils_memory import update_period2farm_and_farm2period_train, update_period2farm_and_farm2period_test
from source.models.memory.mlflow_farm_filtering import FarmCollaborativeFiltering

def run_farm_collaborative_experiment(training_df, validation_df, **params):
    " Run farm-based collaborative filtering experiment "

    # preprocess data for period-based collaborative filtering
    _, farm2period, periodfarm2power_train = update_period2farm_and_farm2period_train(training_df)
    periodfarm2power_validation = update_period2farm_and_farm2period_test(validation_df)

    lst_farms = list(farm2period.keys())

    farm_collaborative_filtering = FarmCollaborativeFiltering(farm2period, periodfarm2power_train, lst_farms, K=params['neighbors'], min_common_periods=params['min_common_periods'])   

    with mlflow.start_run(run_name=params['based_on']):

        _, _, _, _ = farm_collaborative_filtering.compute_similarities()

        # Evaluate the model
        validation_predictions, validation_targets, _, _ = farm_collaborative_filtering.predict(periodfarm2power_validation)
        eval_rmse = farm_collaborative_filtering.compute_rmse(validation_predictions, validation_targets)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Farm-based collaborative filtering ")

        # Log params and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)
        
        # Save model
        mlflow.pyfunc.log_model(artifact_path = params['model'],
                                python_model = farm_collaborative_filtering,
                                #registered_model_name= params['model'] +'-' + params['solver'],
                                )
