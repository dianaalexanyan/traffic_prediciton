import pandas as pd
from datetime import datetime

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data."""
    # Convert 'Time' column to datetime format
    df['Time'] = pd.to_datetime(df['Time'])

    # Extract only the time part in 24-hour format
    df['Time'] = df['Time'].dt.strftime('%H:%M:%S')

    # Ensure the 'Time' column is stored as a string
    df['Time'] = df['Time'].astype(str)

    # Create a new 'Hour_num' column by extracting the hour part
    df['Hour_num'] = pd.to_datetime(df['Time']).dt.hour

    # Convert 'Traffic Situation' to categorical
    easy_traffic = set(df['Traffic Situation'])
    traffic_mapping = {'low': 1, 'normal': 2, 'high': 3, 'heavy': 4}
    for traffic in easy_traffic:
        if traffic not in traffic_mapping.values():
            traffic_mapping[traffic] = traffic_mapping.get(traffic, traffic)
    df['Traffic Situation'] = df['Traffic Situation'].map(traffic_mapping)

    # Convert 'Day of the week' to numerical values
    unique_days = set(df['Day of the week'])
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    for day in unique_days:
        if day not in day_mapping.values():
            day_mapping[day] = day_mapping.get(day, day)
    df['Day of the week'] = df['Day of the week'].map(day_mapping)

    # Combine 'Date' and 'Hour' into a single datetime column
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), format='%d %H:%M:%S', errors='coerce')

    # Extract 'Hour_num' from 'DateTime'
    df['Hour_num'] = df['DateTime'].dt.hour

    # Calculate the average of CarCount by rows for each hour of each day of the week
    df['Avg_CarCount'] = df.groupby(['Day of the week', 'Hour_num'])['CarCount'].transform('mean')

    # Calculate the average of BikeCount by rows for each hour of each day of the week
    df['Avg_BikeCount'] = df.groupby(['Day of the week', 'Hour_num'])['BikeCount'].transform('mean')

    # Calculate the average of BusCount by rows for each hour of each day of the week
    df['Avg_BusCount'] = df.groupby(['Day of the week', 'Hour_num'])['BusCount'].transform('mean')

    # Calculate the average of TruckCount by rows for each hour of each day of the week
    df['Avg_TruckCount'] = df.groupby(['Day of the week', 'Hour_num'])['TruckCount'].transform('mean')

    # Calculate the average of Total by rows for each hour of each day of the week
    df['Avg_Total'] = df.groupby(['Day of the week', 'Hour_num'])['Total'].transform('mean')

    # Drop the new column "DateTime"
    df = df.drop('DateTime', axis=1)

    return df

