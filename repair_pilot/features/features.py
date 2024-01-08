from workalendar.europe import RhinelandPalatinate
from utils.time import to_datetime
import pandas as pd
from utils.data import import_csv_as_df, save_df_as_csv
from utils.const import PREPROCESSED_FILE_PATH, FEATURED_FILE_PATH
from pathlib import Path


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
    """
    # Convert dates to date time
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)

    # Create a calendar object for RhinelandPalatinate
    cal = RhinelandPalatinate()

    # Calculate the number of working days
    working_days = cal.get_working_days_delta(start_date, end_date)

    return working_days


def feature_engineering():
    """
    Creates new features.
    """
    if FEATURED_FILE_PATH.exists():
        pass

    else:
        df = import_csv_as_df(PREPROCESSED_FILE_PATH)
        
        # Creation of 2 new features from feature "adresse" ("adresse" can be divded in a "Kundennummer" and a "Ansprechpartner")
        df[['Kundennummer', 'Ansprechpartner']] = df['adresse'].astype(str).str.split('.', expand=True)
        
        # Define the date columns and their abbreviations
        date_columns = ['waein', 'repstart', 'repkv', 'kv', 'auftrag', 'repend', 'wausgang']

        # Calculate working days between consecutive date columns
        for i in range(len(date_columns) - 1):
            start_col = f'datum_{date_columns[i]}'
            end_col = f'datum_{date_columns[i + 1]}'
            result_col = f'{date_columns[i]}_bis_{date_columns[i + 1]}_arb'

            df[result_col] = df.apply(
                lambda row: calculate_working_days(df[start_col].iloc[row.name], df[end_col].iloc[row.name]), axis=1
            )

        # Calculate the "Durchlaufzeit" 
        df["Durchlaufzeit_arb"] = df[[f'{col}_bis_{date_columns[i+1]}_arb' for i, col in enumerate(date_columns[:-1])]].sum(axis=1)

        # Calculate 'Arbeitseinheit' grouped by 'SERVICEAUFTRAG' for 'KK002' and 'KK060'
        df_service_work_units_sum = df[df['ARTIKEL'].isin(['KK002', 'KK060'])].groupby('SERVICEAUFTRAG')['MENGE'].sum().reset_index(name='Arbeitseinheit')

        # Merge the calculated 'Arbeitseinheit' feature into the original DataFrame
        df_features = pd.merge(df, df_service_work_units_sum, on='SERVICEAUFTRAG', how='left')

        # Save data
        save_df_as_csv(df_features, FEATURED_FILE_PATH)



    

