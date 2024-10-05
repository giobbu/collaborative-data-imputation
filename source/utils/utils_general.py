import pandas as pd 

def read_excel_file(file_path):
    """
    Reads an Excel file and skips the first two rows.
    """
    # Read the Excel file and skip rows 1 and 2
    df = pd.read_excel(file_path, skiprows=[1, 2])
    # Rename columns and set 'datetimeId' as the index
    df.columns = ['datetimeId'] + list(df.columns)[1:]
    df = df.set_index('datetimeId')
    return df


def read_nordpool_csv(file_path):
    """
    Preprocesses a CSV file containing Date and Hours columns.
    """
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


