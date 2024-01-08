import numpy as np
import pandas as pd
from pathlib import Path
from utils.const import PREPROCESSED_FILE_PATH, ROOT_DIR
from utils.data import save_df_as_csv, import_csv_as_df


def clean(df_kopf: pd.DataFrame, df_pos: pd.DataFrame) -> pd.DataFrame: 
    """
    Cleans data.

    Args:
        df_kopf (pandas.DataFrame): Pandas DataFrame for raw "Kopf" data.
        df_pos (pandas.DataFrame): Pandas DataFrame for raw "Pos" data.

    Returns:
        df_kopf (pandas.Dataframe): A DataFrame object containing the cleaned "Kopf" data.  
        df_pos (pandas.Dataframe): A DataFrame object containing the cleaned "Pos" data.  
    """
    # Copy kopf and pos data
    df_kopf = df_kopf.copy()
    df_pos = df_pos.copy()

    # Drops all NaN values
    df_kopf = df_kopf.dropna()
    df_pos = df_pos.dropna()
    
    # Reformation of the feature "adresse" of "df_kopf" by removing the first Point "." of each string
    df_kopf['adresse'] = df_kopf['adresse'].apply(lambda x: x[:2] + x[3:])

    # Rename, convert types, and format values in columns
    df_kopf = df_kopf.rename({'serviceauftrag': 'SERVICEAUFTRAG'}, axis=1)
    df_pos['MENGE'] = df_pos['MENGE'].apply(lambda x: float(x.replace(',', '.')))

    return df_kopf, df_pos


def merge(df_kopf: pd.DataFrame, df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the two pandas DataFrames "Kopf" and "Pos" based on their shared "SERVICEAUFTRAG" feature and sorts the resulting DataFrame.

    Args:
        df_kopf (pandas.DataFrame): The first DataFrame to merge, containing a feature "SERVICEAUFTRAG".
        df_pos (pandas.DataFrame): The second DataFrame to merge, also containing a feature "SERVICEAUFTRAG".

    Returns:
        df_merged (pandas.DataFrame): A new pandas DataFrame resulting from the merge of "df_kopf" and "df_pos", sorted by 
        "SERVICEAUFTRAG", "POSITION", and "datum_waein".
    """
    # Merge "Kopf" data (df_kopf) and "Pos" data (df_pos) with the feature "SERVICEAUFTRAG" to a new dataframe
    df_merged = pd.merge(df_kopf, df_pos, on="SERVICEAUFTRAG")

    # Sorting
    df_merged = df_merged.sort_values(by=['SERVICEAUFTRAG', 'POSITION', 'datum_waein'], ignore_index=True)

    return df_merged


def filter(df_new_ft: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the input DataFrame by dropping rows based on certain conditions and selecting only relevant columns.

    Args:
        df_new_ft (pandas.DataFrame): The input DataFrame that contains the merged "Kopf" and "Pos" dataframe (df_merged).

    Returns:
        df_filtered (pandas.DataFrame): The filtered DataFrame after applying the filters to the merged "Kopf" and "Pos" dataframe (df_merged).
    """
    # Copy Dataframe with new features
    df_filtered = df_new_ft.copy()

    # Remove all "ARTIKEL" that dont start with 'K0' --> e. g. not repair parts
    df_filtered = df_filtered[df_filtered['ARTIKEL'].apply(lambda x: x.startswith('K0'))]

    # Drop datapoints in feature "ARTIKEL" that match with "artikel"
    df_filtered = df_filtered[~df_filtered['ARTIKEL'].isin(df_filtered['artikel'].unique())]

    return df_filtered


def preprocessing():

    """Performs all preprocessing steps."""
    
    if PREPROCESSED_FILE_PATH.exists():
        pass

    else:
        # Import data
        df_kopf = import_csv_as_df(Path.joinpath(ROOT_DIR, 'data', 'raw', 'kopf', 'kopf.csv'))
        df_pos = import_csv_as_df(Path.joinpath(ROOT_DIR, 'data', 'raw', 'pos', 'pos.csv'))
        
        # Clean Data
        df_kopf, df_pos = clean(df_kopf, df_pos)
        
        # Merge Data
        df_merged = merge(df_kopf, df_pos)

        # Filter data
        df_filtered = filter(df_merged)
        
        # Save df_filtered as a CSV file (will be later used for Data Analysis and Clustering)
        save_df_as_csv(df_filtered, PREPROCESSED_FILE_PATH)






