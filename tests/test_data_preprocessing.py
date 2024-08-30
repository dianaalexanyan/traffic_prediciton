# tests/test_data_preprocessing.py
import pandas as pd
from src.traffic_prediction.data_processing.data_preprocessing import load_data, preprocess_data


def test_load_data():
    df = load_data('data/Traffic.csv')
    assert isinstance(df, pd.DataFrame)


def test_preprocess_data():
    # Load the data
    df = load_data('data/Traffic.csv')

    # Preprocess the data
    df_processed = preprocess_data(df)

    # Check if the resulting DataFrame has the expected columns
    expected_columns = ['Hour_num', 'Avg_Total', 'Avg_CarCount', 'Avg_BikeCount', 'Avg_BusCount', 'Avg_TruckCount',
                        'Day of the week', 'Traffic Situation']
    assert all(col in df_processed.columns for col in expected_columns), "Columns mismatch after preprocessing"

    # Check if there are no missing values in the processed DataFrame
    assert df_processed.isnull().sum().sum() == 0, "Missing values present in the processed data"

    # Add more specific tests based on your preprocessing logic

    # For example, check if 'Hour_num' values are within a valid range
    assert df_processed['Hour_num'].between(0, 23).all(), "'Hour_num' values are not within the valid range"

    # For categorical columns, check if they have the expected unique values
    expected_unique_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    assert set(df_processed['Day of the week'].unique()) == set(
        expected_unique_days), "Unexpected unique values in 'Day of the week'"

    expected_unique_traffic_situations = ['low', 'normal', 'high', 'heavy']
    assert set(df_processed['Traffic Situation'].unique()) == set(
        expected_unique_traffic_situations), "Unexpected unique values in 'Traffic Situation'"
