import pickle
import pandas as pd
from loguru import logger

def save_results_to_file(train_df, valid_df, id2datetime_mapping, id2farm_mapping, max_mapping_train, min_mapping_train, file_path):
    " Save results to a file using pickle.dump(). "

    assert isinstance(train_df, pd.DataFrame), "Input train_df must be a pandas DataFrame."
    assert isinstance(valid_df, pd.DataFrame), "Input valid_df must be a pandas DataFrame."
    assert isinstance(id2datetime_mapping, dict), "Input id2datetime_mapping must be a dictionary."
    assert isinstance(id2farm_mapping, dict), "Input id2farm_mapping must be a dictionary."
    assert isinstance(max_mapping_train, float), "Input max_mapping_train must be a float."
    assert isinstance(min_mapping_train, float), "Input min_mapping_train must be a float."
    assert isinstance(file_path, str), "Input file_path must be a string."

    # Create a dictionary to store the results
    logger.info("Saving results to a file...")
    results = {
        'train': train_df,
        'val': valid_df,
        'id2datetime_mapping': id2datetime_mapping,
        'id2farm_mapping': id2farm_mapping,
        'max_mapping_train' : max_mapping_train,
        'min_mapping_train' : min_mapping_train
    }
    # Save results to a file using pickle.dump()
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)
    logger.info(f"Results saved to {file_path} successfully.")



def _load_results(path_list):
    " Load results from a list of file paths using pickle.load(). "
    try:
        loaded_data = []
        for path in path_list:
            with open(path, 'rb') as file:
                loaded_data.append(pickle.load(file))
        return loaded_data
    except Exception as e:
        print(f"Error occurred during data loading: {e}")
        return None


def load_merge_results(paths, method_names, dataset='val'):
    " Load and merge results from multiple files using pickle.load(). "

    assert isinstance(paths, list), "Input paths must be a list."
    assert isinstance(method_names, list), "Input method_names must be a list."
    assert isinstance(dataset, str), "Input dataset must be a string."

    # Load results from multiple files
    data_dicts = _load_results(paths)

    if len(data_dicts) != len(method_names) + 1:
        print("Error: Mismatch in the number of dictionaries and method names provided")
        return None

    # Merge results from multiple files
    df = data_dicts[0][dataset]
    pred_key_in = f'pred_{dataset}'
    for i, method_name in enumerate(method_names):
        if method_name == paths[i+1].split('.')[0].split('_')[-1]:
            pred_key_out = f'pred_{method_name}_{dataset}_z'
            df[pred_key_out] = data_dicts[i+1][pred_key_in]
        else:
            print('Error in method_names')
    # Map periodId and farmId to datetime and farm names
    if 'id2datetime_mapping' in data_dicts[0]:
        df['periodId'] = df['periodId'].map(data_dicts[0]['id2datetime_mapping'])
    if 'id2farm_mapping' in data_dicts[0]:
        df['farmId'] = df['farmId'].map(data_dicts[0]['id2farm_mapping'])
    if dataset == 'train':
        max_mapping_train = data_dicts[0]['max_mapping_train']
        min_mapping_train = data_dicts[0]['min_mapping_train']
        df['power_train_z'] = df['power_z']
        return df, max_mapping_train, min_mapping_train
    df['power_val_z'] = df['power_z']
    return df


def create_farm_dataframe(df, training_df, validation_df, farm_name):
    " Create a DataFrame for a specific farm. "

    assert isinstance(df, pd.DataFrame), "Input df must be a pandas DataFrame."
    assert isinstance(training_df, pd.DataFrame), "Input training_df must be a pandas DataFrame."
    assert isinstance(validation_df, pd.DataFrame), "Input validation_df must be a pandas DataFrame."
    assert isinstance(farm_name, str), "Input farm_name must be a string."

    min_timestamp = df.index.min()
    max_timestamp = df.index.max()
    timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq='1h')
    df_ = pd.DataFrame({'periodId': timestamps}).set_index('periodId')
    df_farm_train = training_df[training_df.farmId == farm_name].set_index('periodId').drop("farmId", axis=1)
    df_farm_val = validation_df[validation_df.farmId == farm_name].set_index('periodId').drop("farmId", axis=1)
    df_farm = df_.join(df_farm_train, on='periodId').join(df_farm_val, on='periodId').reset_index()
    return df_farm


def plot_results(df_farm, start_, end_, y_list_power, y_list_train, y_list_als, y_list_sgd, y_list_naive, y_list_avg):
    " Plot results for a specific farm. "

    assert isinstance(df_farm, pd.DataFrame), "Input df_farm must be a pandas DataFrame."
    assert isinstance(start_, int), "Input start_ must be an integer."
    assert isinstance(end_, int), "Input end_ must be an integer."
    assert isinstance(y_list_power, list), "Input y_list_power must be a list."
    assert isinstance(y_list_train, list), "Input y_list_train must be a list."
    assert isinstance(y_list_als, list), "Input y_list_als must be a list."
    assert isinstance(y_list_sgd, list), "Input y_list_sgd must be a list."
    assert isinstance(y_list_naive, list), "Input y_list_naive must be a list."
    assert isinstance(y_list_avg, list), "Input y_list_avg must be a list."

    df_sorted = df_farm.iloc[start_:end_].sort_values(by="periodId")

    # Plot results
    df_sorted.plot(x="periodId", y=y_list_power, colormap='Spectral', figsize=(22, 5))
    df_sorted.plot(x="periodId", y=y_list_train, colormap='Paired', figsize=(22, 5))
    df_sorted.sort_values(by="periodId").plot(x="periodId", y=y_list_als, figsize=(22, 5))
    df_sorted.sort_values(by="periodId").plot(x="periodId", y=y_list_sgd, figsize=(22, 5))
    df_sorted.sort_values(by="periodId").plot(x="periodId", y=y_list_naive, figsize=(22, 5))
    df_sorted.sort_values(by="periodId").plot(x="periodId", y=y_list_avg, figsize=(22, 5))
