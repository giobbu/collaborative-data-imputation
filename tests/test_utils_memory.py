import pytest
import pandas as pd
from source.utils.utils_memory import update_period2farm_and_farm2period_train

# Fixture to provide a valid sample DataFrame
@pytest.fixture
def sample_df():
    " Provide a sample DataFrame with three columns. "
    data = {
        'periodId': [1, 2, 1, 3],
        'farmId': ['A', 'B', 'C', 'A'],
        'power_z': [0.5, 1.2, 0.7, 0.8]
    }
    return pd.DataFrame(data)

# Fixture to provide a sample DataFrame with missing columns
@pytest.fixture
def sample_df_missing_columns():
    " Provide a sample DataFrame with missing columns. "
    data = {'periodId': [1, 2, 1],
            # 'farmId' column is intentionally missing
            'power_z': [0.5, 1.2, 0.7]
            }
    return pd.DataFrame(data)

# Test case: valid DataFrame
def test_update_period2farm_and_farm2period_train_valid(sample_df):
    " Test if the dictionaries are created correctly for a valid DataFrame. "
    # Call the function with a valid DataFrame
    period2farm, farm2period, periodfarm2power_train = update_period2farm_and_farm2period_train(sample_df)
    # Assert that the dictionaries are created correctly
    assert period2farm == {1: ['A', 'C'], 2: ['B'], 3: ['A']}, "period2farm dictionary is not correct."
    assert farm2period == {'A': [1, 3], 'B': [2], 'C': [1]}, "farm2period dictionary is not correct."
    assert periodfarm2power_train == {(1, 'A'): 0.5, (2, 'B'): 1.2, (1, 'C'): 0.7, (3, 'A'): 0.8}, "periodfarm2power_train dictionary is not correct."

# Test case: invalid input (not a DataFrame)
def test_update_period2farm_and_farm2period_train_invalid_input():
    " Test if an assertion error is raised for an invalid input. "
    with pytest.raises(AssertionError, match="Input df must be a pandas DataFrame."):
        update_period2farm_and_farm2period_train([1, 2, 3])


# Test case: missing columns in DataFrame
def test_update_period2farm_and_farm2period_train_missing_columns(sample_df_missing_columns):
    " Test if an assertion error is raised for a DataFrame missing required columns. "
    # Call the function with a DataFrame missing required columns
    with pytest.raises(AssertionError, match="DataFrame df must contain columns 'periodId', 'farmId', and 'power_z'."):
        update_period2farm_and_farm2period_train(sample_df_missing_columns)

