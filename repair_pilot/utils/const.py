from utils.file import get_parent_folder
from pathlib import Path


#Paths
ROOT_DIR = get_parent_folder("repair-pilot")
PREPROCESSED_FILE_PATH = Path.joinpath(ROOT_DIR, 'repair_pilot','data', 'preprocessed', 'kopf_pos_preprocessed.csv')
FEATURED_FILE_PATH = Path.joinpath(ROOT_DIR, 'repair_pilot', 'data', 'featured', 'features.csv')
RESULTS_PATH = Path.joinpath(ROOT_DIR,'repair_pilot', 'results')