import mlflow
from source.process.process_general import melt_dataframe
from source.utils.utils_memory import update_period2farm_and_farm2period_train, update_period2farm_and_farm2period_test, create_lag_features
from source.models.memory.mlflow_lag_farm_filtering import LagFarmDataImputation 
from loguru import logger

def run_lag_farm_collaborative_experiment(training_df, validation_df, **params):
    " Run lag-farm-based collaborative filtering experiment "

    # create dataset with lags
    nr_lags = params['nr_lags']
    lookup = params['lookup']

    logger.info(f'Lookup scheme - {lookup}')
    logger.info(f'Nr of Lags: {nr_lags}')

    training_df_unmelted = training_df.pivot_table(index='periodId', columns='farmId', values='power_z')
    training_df_unmelted_lags, lst_farms, _ = create_lag_features(training_df_unmelted, params['nr_lags'], params['lookup'])
    training_df_lags, _  = melt_dataframe(training_df_unmelted_lags, id_vars_='periodId', var_name_='farmId', value_name_='power_z')    
    logger.success('Lags successfully created')

    # preprocess data for period-based collaborative filtering
    _, farm2period, periodfarm2power_train = update_period2farm_and_farm2period_train(training_df_lags)
    periodfarm2power_validation = update_period2farm_and_farm2period_test(validation_df)
    logger.success('Dictionaries successfully created')

    lst_farms_lags = list(farm2period.keys())
    other_farms = params['other_farms']
    nr_ts = len(lst_farms_lags)
    logger.info(f'Total Nr of Timeseries: {nr_ts}')
    logger.info(f'Apply Info from other farms - {other_farms}')

    farm_collaborative_filtering = LagFarmDataImputation(farm2period, periodfarm2power_train, lst_farms, lst_farms_lags, K=params['neighbors'], min_common_periods=params['min_common_periods'], other_farms=params['other_farms'])   

    with mlflow.start_run(run_name=params['based_on']):

        _, _, _, _ = farm_collaborative_filtering.compute_similarities()

        # Evaluate the model
        validation_predictions, validation_targets, _, _ = farm_collaborative_filtering.predict(periodfarm2power_validation)
        eval_rmse = farm_collaborative_filtering.compute_rmse(validation_predictions, validation_targets)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Lag-Farm-based collaborative filtering ")

        # Log params and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)
        
        # Save model
        mlflow.pyfunc.log_model(artifact_path = params['model'],
                                python_model = farm_collaborative_filtering,
                                #registered_model_name= params['model'] +'-' + params['solver'],
                                )
