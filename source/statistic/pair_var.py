from loguru import logger
from joblib import Parallel, delayed

def I_jk(df, target_column, reference_column):
    """
    Calculate the proportion of usable cases for imputing values in the target column 
    from the reference column using the Ijk method.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - target_column: Name of the target column for imputation.
    - reference_column: Name of the reference column for imputation.

    Returns:
    - proportion: Proportion of usable cases for imputing values in the target column 
                    from the reference column using the Ijk method.
    """
    try:
        # Count the number of pairs (target_column, reference_column) with target_column missing and reference_column observed
        missing_count = df[target_column].isnull() & ~df[reference_column].isnull()
        # Count the total number of missing cases in target_column
        total_observed = df[target_column].isnull()
        # Calculate the proportion of usable cases
        proportion = missing_count.sum() / total_observed.sum() if total_observed.sum() != 0 else 0
    except Exception as e:
        logger.error(f"Error in calculating proportion for {target_column} and {reference_column}: {e}")
        proportion = None
    return proportion

def O_jk(df, target_column, reference_column):
    """
    Calculate the proportion of outbound for imputing values in the target column 
    from the reference column using the Ojk method.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - target_column: Name of the target column for imputation.
    - reference_column: Name of the reference column for imputation.

    Returns:
    - proportion: Proportion of outbound for imputing values in the target column 
                    from the reference column using the Ojk method.
    """
    try:
        # Count the number of pairs (target_column, reference_column) with target_column missing and reference_column observed
        missing_count = ~df[target_column].isnull() & df[reference_column].isnull()
        # Count the total number of missing cases in target_column
        total_observed = ~df[target_column].isnull()
        # Calculate the proportion
        proportion = missing_count.sum() / total_observed.sum() if total_observed.sum() != 0 else 0
    except Exception as e:
        logger.error(f"Error in calculating proportion for {target_column} and {reference_column}: {e}")
        proportion = None
    return proportion

def invalid_op():
    raise Exception("Invalid Statistic")

def missing_data_proportion(df, target_columns, reference_columns, n_jobs=-1, method="Ijk"):
    """
    Calculate the proportion  for imputing values in the target columns 
    from the reference columns using the specified method.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - target_columns: List of target column names for imputation.
    - reference_columns: List of reference column names for imputation.
    - n_jobs: Number of jobs to run in parallel. Default is -1, which means using all processors.
    - method: Method for calculating the proportion of missing data. Default is "Ijk".

    Returns:
    - proportions: Nested dictionary containing the proportion for each pair 
                    of target and reference columns.
    """
    logger.info("______________- Statistic: " + method + " -______________")
    ops = {
        "Ijk": I_jk,
        "Ojk": O_jk
    }
    statistic = ops.get(method, invalid_op)
    def calculate_proportions(target_column, reference_column):
        proportions[target_column][reference_column] = statistic(df, target_column, reference_column)
    try:
        proportions = {target_column: {reference_column: 0 for reference_column in reference_columns} for target_column in target_columns}
        Parallel(n_jobs=n_jobs, prefer="threads")(delayed(calculate_proportions)(target_column, reference_column) 
                                                                    for target_column in target_columns 
                                                                    for reference_column in reference_columns)
    except Exception as e:
        logger.error(f"Error occurred during parallel computation: {e}")
    return proportions
