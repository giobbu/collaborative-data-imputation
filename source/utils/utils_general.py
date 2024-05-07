import pandas as pd 
import numpy as np


def read_excel_file(file_path):
    """
    Reads an Excel file and skips the first two rows.

    Parameters:
    - file_path (str): The path to the Excel file.

    Returns:
    - DataFrame: The DataFrame read from the Excel file with 'datetimeId' as the index.

    If an error occurs during reading, None is returned.
    """
    try:
        # Read the Excel file and skip rows 1 and 2
        df = pd.read_excel(file_path, skiprows=[1, 2])
        
        # Rename columns and set 'datetimeId' as the index
        df.columns = ['datetimeId'] + list(df.columns)[1:]
        df = df.set_index('datetimeId')
        
        return df
    except Exception as e:
        print(f"Error occurred while reading Excel file: {e}")
        return None


def read_nordpool_csv(file_path):
    """
    Preprocesses a CSV file containing Date and Hours columns.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The preprocessed DataFrame with periodId as the index.

    If an error occurs during preprocessing, None is returned.
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Convert Date column to datetime.date object
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y').dt.date.astype(str)

        # Extract hour from Hours column and convert to time object
        df['Hours'] = pd.to_datetime(df['Hours'].str[:2], format='%H').dt.time.astype(str)

        # Combine Date and Hours columns into periodId
        df['periodId'] = pd.to_datetime(df['Date'] + ' ' + df['Hours'])

        # Set periodId as index and drop Date and Hours columns
        df.set_index('periodId', inplace=True)
        df.drop(['Date', 'Hours'], axis=1, inplace=True)

        return df

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
