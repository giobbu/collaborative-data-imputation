import pandas as pd
import numpy as np

def update_period2farm_and_farm2period_train(df):
    """
    Update dictionaries to map periods to farms and farms to periods for training data.
    """

    assert isinstance(df, pd.DataFrame), "Input df must be a pandas DataFrame."
    assert 'periodId' in df.columns and 'farmId' in df.columns and 'power_z' in df.columns, "DataFrame df must contain columns 'periodId', 'farmId', and 'power_z'."

    # Initialize dictionaries
    period2farm = {}
    farm2period = {}
    periodfarm2power_train = {}
    for _, row in df.iterrows():
        # Update period2farm dictionary
        i = row['periodId']
        j = row['farmId']
        if i not in period2farm:
            period2farm[i] = [j]  # Initialize list with farmId
        else:
            period2farm[i].append(j)  # Append farmId to list
        # Update farm2period dictionary
        if j not in farm2period:
            farm2period[j] = [i]  # Initialize list with periodId
        else:
            farm2period[j].append(i)  # Append periodId to list
        # Update periodfarm2power_train dictionary
        periodfarm2power_train[(i,j)] = row.power_z
    return period2farm, farm2period, periodfarm2power_train


def update_period2farm_and_farm2period_test(df):
    """
    Update dictionary to map periodfarm to power_z values for test data.
    """
    
    assert isinstance(df, pd.DataFrame), "Input df must be a pandas DataFrame."
    assert 'periodId' in df.columns and 'farmId' in df.columns and 'power_z' in df.columns, "DataFrame df must contain columns 'periodId', 'farmId', and 'power_z'."

    # Initialize dictionary
    periodfarm2power_test = {}
    for _, row in df.iterrows():
        i = row['periodId']  # periodId
        j = row['farmId']  # farmId
        periodfarm2power_test[(i, j)] = row['power_z']  # Update dictionary
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