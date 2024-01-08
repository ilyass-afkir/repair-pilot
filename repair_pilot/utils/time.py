# Import Libraries
from workalendar.europe import RhinelandPalatinate
import pandas as pd


# Functions
def to_datetime(time_string: str) -> pd.Timestamp:
    """
    Converts a string representing a date or time to a pandas datetime object.
    
    Args:
        time_string (str): A string representing a date or time in one of several possible formats.
    
    Returns:
        pd.Timestamp: A pandas Timestamp object representing the parsed date or time.
    
    Raises:
        ValueError: If the input string cannot be parsed into a valid date or time format.
    """
    
    # Loop through date_formats. The list can be extend by more date_formats.
    for date_format in ['%d-%m-%Y', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d.%m.%Y %H:%M']:
        
        try:
            date_time = pd.to_datetime(time_string, format=date_format)
            return date_time
        
        except ValueError:
            pass
        

def calculate_working_days(start_date: str, end_date: str) -> int:
    
    """
    Calculate the number of working days between two dates, excluding weekends and public holidays.

    Args:
        start_date (str): The starting date, in string format in one of the following formats:
                          '%d-%m-%Y', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d.%m.%Y %H:%M', '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d'.
        end_date (str): The ending date, in string format in one of the following formats:
                        '%d-%m-%Y', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d.%m.%Y %H:%M', '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d'.

    Returns:
        number_of_workingdays (int): The number of working days between the start and end dates.

    Raises:
        ValueError: If either the start or end date is not in the correct format.
    
    """

    # Convert dates to date time
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)

    # Create a calendar object for RhinelandPalatinate
    cal = RhinelandPalatinate()

    # Calculate the number of working days
    working_days = cal.get_working_days_delta(start_date, end_date)

    return working_days


def validate_dateformat(df: pd.DataFrame, date_feature: str, date_str: str, date_formats: list[str]) -> bool:
    
    """
    Check if a date string is in the correct format, and if it is within the range of dates in a Pandas DataFrame.
    
    Args:
        date_str (str): A string representing the date to be checked.
        date_formats (list): A list of date formats (in strings) to be used to parse the date string.
        date_feature (str): The name of the date feature in the DataFrame to be checked.
        df (pd.DataFrame): A Pandas DataFrame containing the date feature to be checked.
        
    Returns:
        bool: Returns True if the date is in the correct format and within the range of dates in the DataFrame.
        
    Raises:
        ValueError: If the date string is not in any of the specified formats, or if it is not within the range of dates in the DataFrame.
    
    Example:
        >>> validate_dateformat(df, "datum_wausgang", "30.06.2018", ["%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"])

    """
    
    # Check if "date_str" is in the correct format
    if len(date_str) not in [8, 10]:
        raise ValueError(f"Warning: The (start or end) date {date_str} is not in the correct format.\nPlease use one of the following formats: {date_formats}.")
    
  
    for date_format in date_formats:
        try:
            pd.to_datetime(date_str, errors='raise', format=date_format)
            break
        except ValueError:
            pass
    else:
        raise ValueError(f"Warning: The date {date_str} is not in the correct format.\nPlease use one of the following formats: {date_formats}")
  
    # Convert "date_str" and "date_feature" to datetime
    df[f"{date_feature}_dt"] = df[date_feature].apply(lambda x: to_datetime(x))
    date_str_dt = to_datetime(date_str)
    
    min_date = df[f"{date_feature}_dt"].min()
    max_date = df[f"{date_feature}_dt"].max()
    # Check if "date_str" is in the data.
    try:
        if (date_str_dt < min_date) or (date_str_dt > max_date):
            raise ValueError(f"Warning: The (start or end) date {date_str} is not found in the feature {date_feature}.\nThe Data will be filtered automatically by the earliest start date {min_date} or latest end date {max_date} of feature {date_feature}.")
    except ValueError as e:
        print(e)


    return True


def time_filter(df: pd.DataFrame, start_date: str, end_date: str, date_feature: str) -> pd.DataFrame:

    """
    Filters the preprocessed data by product type and a date range by using the specified date feature.

    Args:
        df (pd.DataFrame): The input DataFrame (preprocessed data) to filter.
        product_type (str): The desired product type to filter by.
        date_feature (str): The name of the date feature in the input DataFrame to use for filtering.
        start_date (str): The starting date of the desired date range, in one of the following formats:
                          '%d-%m-%Y', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d'
        end_date (str): The ending date of the desired date range, in one of the following formats:
                        '%d-%m-%Y', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d'

    Returns:
        df_filtered (pd.DataFrame): A new DataFrame that contains only the rows with the desired product type and within the specified date range.

    Raises:
        ValueError: If the start_date or end_date is not in a valid format.
        ValueError: If the start_date or end_date is not found in the specified date feature of the input DataFrame.

    Examples:
        >>> df = filter_data(df, "MM", "datum_wausgang", "01.01.2018", "31.12.2018")
        >>> df = filter_data(df, "MAK-18", "datum_wausgang", "01-05-2018", "31-05-2018")
        >>> df = filter_data(df, "MM", "datum_wausgang", "12/07/2015", "18/07/2019")
    
    """
    
    # Define "date_formats"
    date_formats = ["%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"]
    
    # Check if "start_date" is in the correct format and in the data
    validate_dateformat(df, date_feature, start_date, date_formats)
         
    # Check if "end_date" is in the correct format and in the data
    validate_dateformat(df, date_feature, end_date, date_formats)
    
    # Convert "date_str" and "date_feature" to datetime
    df[f"{date_feature}_dt"] = df[date_feature].apply(lambda x: to_datetime(x))
    start_date_dt = to_datetime(start_date)
    end_date_dt = to_datetime(end_date)

    # Filter by the defined "start_date" and "end_date"
    df_filtered = df[(df[f"{date_feature}_dt"] >= start_date_dt) & (df[f"{date_feature}_dt"] <= end_date_dt)]

    return df_filtered
