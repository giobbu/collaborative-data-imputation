import mlflow
from source.utils.utils_memory import update_period2farm_and_farm2period_train, update_period2farm_and_farm2period_test
from source.models.memory.mlflow_period_filtering import PeriodCollaborativeFiltering

def run_period_collaborative_experiment(training_df, validation_df, **params):
    " Run period-based collaborative filtering experiment "

    # preprocess data for period-based collaborative filtering
    period2farm, _, periodfarm2power_train = update_period2farm_and_farm2period_train(training_df)
    periodfarm2power_validation = update_period2farm_and_farm2period_test(validation_df)

    lst_periods = list(period2farm.keys())

    period_collaborative_filtering = PeriodCollaborativeFiltering(period2farm, periodfarm2power_train, lst_periods, K=params['neighbors'], min_common_farms=params['min_common_farms'])   

    with mlflow.start_run(run_name=params['based_on']):

        _, _, _, _ = period_collaborative_filtering.compute_period_similarities()

        # Evaluate the model
        validation_predictions, validation_targets, _, _ = period_collaborative_filtering.predict(periodfarm2power_validation)
        eval_rmse = period_collaborative_filtering.compute_rmse(validation_predictions, validation_targets)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Period-based collaborative filtering ")

        # Log params and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)
        
        # Save model
        mlflow.pyfunc.log_model(artifact_path = params['model'],
                                python_model = period_collaborative_filtering,
                                #registered_model_name= params['model'] +'-' + params['solver'],
                                )
