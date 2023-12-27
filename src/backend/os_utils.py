# Import Libraries
import os, csv, chardet
import pandas as pd
from plotly.graph_objs._figure import Figure


#Functions

def get_root_dir():
    """Returns the absolute path of the 'AI_Repair_System' directory"""
    return os.path.abspath(os.path.join(os.path.abspath(__file__).split('AI_Repair_System')[0], 'AI_Repair_System'))


def save_df_as_csv(df: pd.DataFrame, folder_path: str, filename: str) -> bool:
    
    """
    Saves a pandas DataFrame 'df' as a CSV file in the specified folder.

    Args:
        folder_path (str): The path to the folder where the CSV file should be saved.
        filename (str): The name of the CSV file.
        df (pandas.DataFrame): The DataFrame that will be saved as a CSV file.

    Returns:
        bool: True if the figure was successfully saved, False otherwise.

    Example:
        >>> save_df_as_csv(df, '/path/to/folder', 'filename')
    
    """
    
    if os.path.isdir(os.path.join(folder_path)) is False:
        os.mkdir(os.path.join(folder_path))
    
    df.to_csv(os.path.join(folder_path, f'{filename}.csv'), index=False, sep=";")

    return True


def save_figure(fig: Figure, folder_path: str, typ: str, start_date: str, end_date: str) -> bool:
    
    """
    Save a html Plotly figure.

    Args:
        fig (Figure): The Plotly figure to save.
        folder_path (str): The path to the folder where the image will be saved.
        filename (str): The name of the image file.
        format (str): The file format of the image (e.g., 'png', 'jpeg', 'svg').

    Returns:
        bool: True if the figure was successfully saved, False otherwise.

    """

    if os.path.isdir(os.path.join(folder_path)) is False:
        os.mkdir(os.path.join(folder_path))
    
    fig.write_html(os.path.join(folder_path, f"{typ}_{start_date}_{end_date}.html"))
    
    return True


def import_raw_data(folder_path: str) -> pd.DataFrame:
    
    """
    Returns a Pandas DataFrame object containing the raw "Kopf" or "Pos" data from a CSV file.

    Args:
        folder_path (str): The path to the folder containing the CSV file of the raw kopf or pos data.
                           
    Raises:
        ValueError: If the folder contains more than one CSV file, a ValueError will be raised.
        ValueError: If the file in the folder is not a CSV file, a ValueError will be raised.
    
    Returns:
        df (pandas.DataFrame): A DataFrame object containing the raw kopf or pos data. 
    
    Example:
        >>> df_kopf = import_raw_data("path/to/kopffolder")
        >>> df_pos = import_raw_data("path/to/posfolder")
    
    """

    # Sets the path to the kopf or pos folder
    folder_path = folder_path
    
    # Check if folder contains only one file
    file_names = os.listdir(folder_path)
    if len(file_names) != 1:
        raise ValueError(f"Folder '{folder_path}' must contain exactly one file.")
    
    # Check if the file is a CSV
    file_name = file_names[0]
    if not file_name.endswith('.csv'):
        raise ValueError(f"File '{file_name}' in folder '{folder_path}' is not a CSV file.")
    
    # Detect the encoding of the CSV file
    with open(os.path.join(folder_path, file_name), "rb") as f:
        data = f.read()
        result = chardet.detect(data)
        encoding = result["encoding"]
    
    # Detect the seperator of the CSV file
    with open(os.path.join(folder_path, file_name), 'r') as file:
        sample = file.read(1024)  # Read a sample of the file
    separator = csv.Sniffer().sniff(sample).delimiter
            
    # Read the CSV File
    df = pd.read_csv(os.path.join(folder_path, file_name), encoding=encoding, sep=separator)
    
    return df


def import_preprocessed_data(folder_path: str) -> pd.DataFrame:
    
    """
    Returns a Pandas DataFrame object containing the raw "preprocessed" data from a CSV file.

    Args:
        folder_path (str): The path to the folder containing the CSV file of the raw kopf or pos data.
                           
    Raises:
        ValueError: If the folder contains more than one CSV file, a ValueError will be raised.
        ValueError: If the file in the folder is not a CSV file, a ValueError will be raised.
    
    Returns:
        df (pandas.DataFrame): A DataFrame object containing the raw kopf or pos data. 
    
    Example:
        >>> df = import_preprocessed_data("path/to/kopffolder")
    
    """

    # Sets the path to the kopf or pos folder
    folder_path = folder_path

    # Check if folder contains only one file
    file_names = os.listdir(folder_path)
    if len(file_names) != 1:
        raise ValueError(f"Folder '{folder_path}' must contain exactly one file.")
    
    # Check if the file is a CSV
    file_name = file_names[0]
    if not file_name.endswith('.csv'):
        raise ValueError(f"File '{file_name}' in folder '{folder_path}' is not a CSV file.")
            
    # Read the CSV File
    df = pd.read_csv(os.path.join(folder_path, file_name), sep=";")
    
    return df
