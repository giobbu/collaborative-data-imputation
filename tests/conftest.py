import pytest
import pandas as pd

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
