import pandas as pd

def load_and_clean_data(file_path):
    """
    Load and clean the data from a given CSV file path.

    This function performs the initial data cleaning by removing unique identifiers, 
    converting dates and times to datetime format, and dropping the 'Invoice ID' column.

    Parameters:
    file_path (str): The path to the CSV file to be loaded.

    Returns:
    pd.DataFrame: The cleaned dataframe.

    """
    # Load the data from the given CSV file path
    df = pd.read_csv(file_path)
    
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # If the 'Time' column exists, convert and extract the hour
    if 'Time' in df.columns:
        # We use format='mixed' or the specific format for 12h: %I:%M %p 
        # The '%I' is for 0-12, '%M' for minutes and '%p' for AM/PM
        time_series = pd.to_datetime(df['Time'], errors='coerce')
        
        # Extract the hour as an integer (Essential for the ML model)
        df['Hour'] = time_series.dt.hour
        
        # Convert the original 'Time' column to a clean string format (HH:MM)
        # This ensures that in your CSV/Dashboard the time is standardized at 24h
        df['Time'] = time_series.dt.strftime('%H:%M')
    
    # Drop the 'Invoice ID' column to avoid overfitting
    if 'Invoice ID' in df.columns:
        df = df.drop(columns=['Invoice ID'])
        
    return df