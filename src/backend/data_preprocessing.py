# Import Libraries
import numpy as np
import pandas as pd
import os
from backend.time_utils import calculate_working_days
from backend.os_utils import get_root_dir, save_df_as_csv, import_raw_data
import time


# Functions

def data_cleaning(df_kopf: pd.DataFrame, df_pos: pd.DataFrame) -> pd.DataFrame:
    
    """
    Performs feature engineering on input DataFrames for the raw "Kopf" and "Pos" data.

    Args:
        df_kopf (pandas.DataFrame): Pandas DataFrame for raw "Kopf" data.
        df_pos (pandas.DataFrame): Pandas DataFrame for raw "Pos" data.

    Returns:
        df_kopf (pandas.Dataframe): A DataFrame object containing the cleaned "Kopf" data.  
        df_pos (pandas.Dataframe): A DataFrame object containing the cleaned "Pos" data.  

    Example:
        >>> df_kopf_cleaned, df_pos_cleaned = data_cleaning(df_kopf, df_pos)
    
    """
    
    # Copy kopf and pos data
    df_kopf = df_kopf.copy()
    df_pos = df_pos.copy()

    # Drops all NaN values
    df_kopf = df_kopf.dropna()
    df_pos = df_pos.dropna()
    
    # Reformation of the feature "adresse" of "df_kopf_clean" by removing the first Point "." of each string
    df_kopf['adresse'] = df_kopf['adresse'].apply(lambda x: x[:2] + x[3:])

    # Rename, convert types, and format values in columns
    df_kopf = df_kopf.rename({'serviceauftrag': 'SERVICEAUFTRAG'}, axis=1)
    df_pos['MENGE'] = df_pos['MENGE'].apply(lambda x: float(x.replace(',', '.')))

    return df_kopf, df_pos


def merge_data(df_kopf: pd.DataFrame, df_pos: pd.DataFrame) -> pd.DataFrame:

    """
    Merges the two pandas DataFrames "Kopf" and "Pos" based on their shared "SERVICEAUFTRAG" feature and sorts the resulting DataFrame.

    Args:
        df_kopf (pandas.DataFrame): The first DataFrame to merge, containing a feature "SERVICEAUFTRAG".
        df_pos (pandas.DataFrame): The second DataFrame to merge, also containing a feature "SERVICEAUFTRAG".

    Returns:
        df_merged (pandas.DataFrame): A new pandas DataFrame resulting from the merge of "df_kopf" and "df_pos", sorted by 
        "SERVICEAUFTRAG", "POSITION", and "datum_waein".

    Raises:
        ValueError: If either input DataFrame does not contain the "SERVICEAUFTRAG" feature.

    Example:
        >>> df_merged = merge_data(df_kopf, df_pos)
       
    """
    
    # Check input features
    if "SERVICEAUFTRAG" not in df_kopf.columns or "SERVICEAUFTRAG" not in df_pos.columns:
        raise ValueError("Both input DataFrames must contain the 'SERVICEAUFTRAG' feature.")

    # Merge "Kopf" data (df_kopf) and "Pos" data (df_pos) with the feature "SERVICEAUFTRAG" to a new dataframe
    df_merged = pd.merge(df_kopf, df_pos, on="SERVICEAUFTRAG")

    # Sorting
    df_merged = df_merged.sort_values(by=['SERVICEAUFTRAG', 'POSITION', 'datum_waein'], ignore_index=True)

    return df_merged


def feature_engineering(df_merged: pd.DataFrame) -> pd.DataFrame:
    
    """
    Performs feature engineering on the merged "Kopf" and "Pos" DataFrame and returns a new DataFrame "df_new_ft" with additional features.
    
    Args:
        df_merged (pd.Dataframe): A pandas DataFrame containing the merged "Kopf" and "Pos" data.
    
    Returns:
        df_new_ft (pd.Dataframe): A new pandas DataFrame with additional features created from the input data.
    
    Example:
        >>> df_new_ft = feature_engineering(df_merged)

    """

    # Copy df_merged
    df_new_ft = df_merged.copy()

    # Creation of 2 new features from feature "adresse" ("adresse" can be divded in a "Kundennummer" and a "Ansprechpartner")
    df_new_ft[['Kundennummer', 'Ansprechpartner']] = df_new_ft['adresse'].str.split('.', expand=True)

    # Working days between "datum_waein" and "datum_repstart"
    df_new_ft['waein_bis_repstart_arbeitstage'] = df_new_ft.apply(lambda row: calculate_working_days(df_new_ft['datum_waein'].iloc[row.name], df_new_ft['datum_repstart'].iloc[row.name]), axis=1)
    
    # Working days between "datum_repstart" and "datum_repkv"
    df_new_ft['repstart_bis_repkv_arbeitstage'] = df_new_ft.apply(lambda row: calculate_working_days(df_new_ft['datum_repstart'].iloc[row.name], df_new_ft['datum_repkv'].iloc[row.name]), axis=1)
    
    # Working days between "datum_repkv" and "datum_kv"
    df_new_ft['repkv_bis_kv_arbeitstage'] = df_new_ft.apply(lambda row: calculate_working_days(df_new_ft['datum_repkv'].iloc[row.name], df_new_ft['datum_kv'].iloc[row.name]), axis=1)
    
    # Working days between "datum_kv" and "datum_auftrag"
    df_new_ft['kv_bis_auftrag_arbeitstage'] = df_new_ft.apply(lambda row: calculate_working_days(df_new_ft['datum_kv'].iloc[row.name], df_new_ft['datum_auftrag'].iloc[row.name]), axis=1)

    # Working days between "datum_auftrag" and "datum_repend"
    df_new_ft['auftrag_bis_repend_arbeitstage'] = df_new_ft.apply(lambda row: calculate_working_days(df_new_ft['datum_auftrag'].iloc[row.name], df_new_ft['datum_repend'].iloc[row.name]), axis=1)
    
    # Working days  between "datum_repend" and "datum_wausgang"
    df_new_ft['repend_bis_wausgang_arbeitstage'] = df_new_ft.apply(lambda row: calculate_working_days(df_new_ft['datum_repend'].iloc[row.name], df_new_ft['datum_wausgang'].iloc[row.name]), axis=1)

    # Working days between "datum_repstart" and "datum_repend"
    df_new_ft['repstart_bis_repend_arbeitstage'] = df_new_ft.apply(lambda row: calculate_working_days(df_new_ft['datum_repstart'].iloc[row.name], df_new_ft['datum_repend'].iloc[row.name]), axis=1)

    # Working days between "datum_waein" and "datum_wausgang" (This is also the "Durchlaufzeit" which can be false at some datapoints because "datum_waein" and "datum_wausgang" are not always the earliest and latest dates)
    df_new_ft['Durchlaufzeit_(waein_bis_wausgang_arbeitstage)'] = df_new_ft.apply(lambda row: calculate_working_days(df_new_ft['datum_waein'].iloc[row.name], df_new_ft['datum_wausgang'].iloc[row.name]), axis=1)
    
    # Calculate the "Durchlaufzeit" alternativly (there should be no errors now) by summing all individual "datum_..._arbeitstage" features because "datum_waein" and "datum_wausgang" are not always the earliest and latest dates
    df_new_ft["Durchlaufzeit"] = df_new_ft['waein_bis_repstart_arbeitstage'] + df_new_ft['repstart_bis_repkv_arbeitstage'] + df_new_ft['repkv_bis_kv_arbeitstage'] +  df_new_ft['kv_bis_auftrag_arbeitstage'] + df_new_ft['auftrag_bis_repend_arbeitstage'] + df_new_ft['repend_bis_wausgang_arbeitstage']

    # Create a new "Arbeitseinheit"/"Unit of work" (KK002, KK060) feature
        # Group the DataFrame by 'repairnumber' and 'sparepartname', and sum the 'sparepartnumber' column
    df_new = df_new_ft.groupby('SERVICEAUFTRAG').apply(lambda x: x[x['ARTIKEL'].isin(['KK002', 'KK060'])]['MENGE'].sum()).reset_index(name='Arbeitseinheit')
        # Merge the dataframes
    df_new_ft = pd.merge(df_new_ft, df_new, on='SERVICEAUFTRAG')
    
    return df_new_ft


def filter_data(df_new_ft: pd.DataFrame) -> pd.DataFrame:

    """
    Filters the input DataFrame by dropping rows based on certain conditions and selecting only relevant columns.

    Args:
        df_new_ft (pandas.DataFrame): The input DataFrame that contains the merged "Kopf" and "Pos" dataframe (df_merged).

    Returns:
        df_filtered (pandas.DataFrame): The filtered DataFrame after applying the filters to the merged "Kopf" and "Pos" dataframe (df_merged).


    Examples:
        >>> df_filtered = filter_data(df)
   
    """

    # Copy Dataframe with new features
    df_filtered = df_new_ft.copy()

    # Remove all "ARTIKEL" that dont start with 'K0' --> e. g. not repair parts
    df_filtered = df_filtered[df_filtered['ARTIKEL'].apply(lambda x: x.startswith('K0'))]

    # Drop datapoints in feature "ARTIKEL" that match with "artikel"
    df_filtered = df_filtered[~df_filtered['ARTIKEL'].isin(df_filtered['artikel'].unique())]

    return df_filtered


def data_preprocessing_main():
    
    """
    This function is the entry point for the program. It uses all prior defined functions.
    
    """
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print(f"\033[1mData preprocessing is now starting\033[0m")
    print("------------------------------------------------------------------------------------")
   
    file_path = os.path.join(get_root_dir(), 'data', 'preprocessed', 'kopf_pos_preprocessed.csv')
    
    if os.path.isfile(file_path):
        print(f"Attention: Data preprocessing was already done!\nStorage location: {file_path}")
    
    else:
        # Start time measuremennt
        start_time = time.time()
        # Import data
        print("Step 1: Importing data")
        df_kopf = import_raw_data(os.path.join(get_root_dir(),'data', 'raw', 'kopf'))
        df_pos = import_raw_data(os.path.join(get_root_dir(),'data', 'raw', 'pos'))
        
        
        # Clean Data
        print("Step 2: Cleaning data")
        df_kopf, df_pos = data_cleaning(df_kopf, df_pos)
        
        
        # Merge Data
        print("Step 3: Merging data")
        df_merged = merge_data(df_kopf, df_pos)
        
        # Feature Engineering
        print("Step 4: Feature engineering")
        df_new_ft = feature_engineering(df_merged)
        
        # Filter data
        print("Step 5: Filtering data")
        df_filtered = filter_data(df_new_ft)
        
        # Save df_filtered as a CSV file (will be later used for Data Analysis and Clustering)
        save_df_as_csv(df_filtered, folder_path=os.path.join(get_root_dir(), 'data', 'preprocessed'), filename='kopf_pos_preprocessed')
        print(f"Step 6: Save results as a CSV file (location: data/preprocessed)")

        # End time measurement
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Step 7: Data preprocessing is completed.Time taken: {time_taken} seconds")



