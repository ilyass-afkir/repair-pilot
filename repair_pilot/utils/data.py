import csv
import chardet
import pandas as pd
from pathlib import Path


def import_csv_as_df(file_path: Path) -> pd.DataFrame:
    """
    Import a CSV file as a Pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    with file_path.open("rb") as f:
        data = f.read()
        result = chardet.detect(data)
        encoding = result["encoding"]

    with file_path.open('r') as file:
        sample = file.read(1024)  
    separator = csv.Sniffer().sniff(sample).delimiter
            
    df = pd.read_csv(file_path, encoding=encoding, sep=separator)
        
    return df
    

def save_df_as_csv(df: pd.DataFrame, storage_path: Path):
    """
    Stores a pandas DataFrame as a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame that will be saved as a CSV file.
        storage_path (str): The relative file path where the CSV file should be saved (including filename and extension).

    Returns:
        bool: True if the CSV file was successfully saved, False otherwise.
    """
    folder = storage_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(storage_path, index=False)
    


    
 
