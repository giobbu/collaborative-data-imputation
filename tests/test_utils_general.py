import pytest
from source.utils.utils_general import read_excel_file, read_nordpool_csv

def test_read_excel_file_invalid_extension():
    " Test if an assertion error is raised for an invalid file extension. "
    with pytest.raises(AssertionError, match="Input file must be an Excel file."):
        read_excel_file('test_file.csv')

def test_read_csv_file_invalid_extension():
    " Test if an assertion error is raised for an invalid file extension. "
    with pytest.raises(AssertionError, match="Input file must be a CSV file."):
        read_nordpool_csv('test_file.xlsx')
