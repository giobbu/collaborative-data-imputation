import pandas as pd
import numpy as np

def update_period2farm_and_farm2period_train(df):
    """
    Update dictionaries to map periods to farms and farms to periods for training data.

    Args:
    - df (pandas.DataFrame): DataFrame containing training data with columns 'periodId', 'farmId', and 'power_z'.

    Returns:
    - period2farm (dict): Dictionary mapping period IDs to lists of corresponding farm IDs.
    - farm2period (dict): Dictionary mapping farm IDs to lists of corresponding period IDs.
    - periodfarm2power_train (dict): Dictionary mapping (periodId, farmId) tuples to power_z values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")
    if 'periodId' not in df.columns or 'farmId' not in df.columns or 'power_z' not in df.columns:
        raise ValueError("DataFrame df must contain columns 'periodId', 'farmId', and 'power_z'.")
    period2farm = {}
    farm2period = {}
    periodfarm2power_train = {}
    for _, row in df.iterrows():
        i = row['periodId']
        j = row['farmId']
        if i not in period2farm:
            period2farm[i] = [j]
        else:
            period2farm[i].append(j)
        if j not in farm2period:
            farm2period[j] = [i]
        else:
            farm2period[j].append(i)
        periodfarm2power_train[(i,j)] = row.power_z
    return period2farm, farm2period, periodfarm2power_train


def update_period2farm_and_farm2period_test(df):
    """
    Update dictionary to map periodfarm to power_z values for test data.

    Args:
    - df (pandas.DataFrame): DataFrame containing test data with columns 'periodId', 'farmId', and 'power_z'.

    Returns:
    - periodfarm2power_test (dict): Dictionary mapping (periodId, farmId) tuples to power_z values for test data.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")
    if 'periodId' not in df.columns or 'farmId' not in df.columns or 'power_z' not in df.columns:
        raise ValueError("DataFrame df must contain columns 'periodId', 'farmId', and 'power_z'.")
    periodfarm2power_test = {}
    for _, row in df.iterrows():
        i = row['periodId']
        j = row['farmId']
        periodfarm2power_test[(i, j)] = row['power_z']
    return periodfarm2power_test

# lags creation

def backwards(df, list_cols, list_lags, nr_lags):
    try:
        df_lags = df.copy()
        # Backwards lag features
        for col in list_cols:
            for lag in list_lags:
                df_lags[f'{col}_lag_-{lag}'] = df[col].shift(lag)
        # Remove rows with NaN values introduced by shifting
        df = df.iloc[nr_lags:, :]
        return df_lags
    except Exception as e:
        print(f"Error occurred in backwards function: {str(e)}")
        return None

def forwards(df, list_cols, list_lags, nr_lags):
    try:
        df_lags = df.copy()
        # Forwards lag features
        for col in list_cols:
            for lag in list_lags:
                df_lags[f'{col}_lag_+{lag}'] = df[col].shift(-lag)
        # Remove rows with NaN values introduced by shifting
        df = df.iloc[:-nr_lags, :]
        return df_lags
    except Exception as e:
        print(f"Error occurred in backwards function: {str(e)}")
        return None

def create_lag_features(df, nr_lags, lookup='both'):
    
    list_cols = list(df.columns)
    list_lags = list(np.arange(nr_lags) + 1)
    
    lookup_functions = {
        'both': [backwards, forwards],
        'backwards': [backwards],
        'forwards': [forwards]
    }
    
    if lookup in lookup_functions:
        for func in lookup_functions[lookup]:
            df = func(df, list_cols, list_lags, nr_lags)
    else:
        raise ValueError('Invalid lookup scheme. Choose from "both", "backwards", or "forwards".')

    # Separate lag features and non-lag features
    list_cols_lags = [name for name in list(df.columns) if 'lag' in str(name)]
    list_cols_farm = [name for name in list(df.columns) if 'lag' not in str(name)]
    
    return df, list_cols_farm, list_cols_lags