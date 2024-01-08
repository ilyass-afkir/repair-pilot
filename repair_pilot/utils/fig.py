from plotly.graph_objs._figure import Figure
from pathlib import Path

def save_plotly_as_html(fig: Figure, storage_path: str):
    """
    Save a Plotly figure as an HTML file.

    Args:
        fig (Figure): The Plotly figure to save.
        storage_path (str): The file path where the HTML file should be saved (including filename and extension).

    Returns:
        bool: True if the figure was successfully saved, False otherwise.
    """

    storage_path = Path(storage_path)
    folder = storage_path.parent
    folder.mkdir(parents=True, exist_ok=True)   
    fig.write_html(storage_path)