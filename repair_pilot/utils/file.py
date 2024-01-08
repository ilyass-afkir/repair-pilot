import pandas as pd
from pathlib import Path


def get_parent_folder(folder_name: str) -> Path:
    """
    Retrieve the absolute path of a parent folder.

    Args:
        folder_name (str): The name of the target folder.

    Returns:
        Path or None: The absolute path of the parent folder if found, None otherwise.
    """
    current_path = Path.cwd()

    while current_path.name != folder_name:
        current_path = current_path.parent

    return current_path



    

