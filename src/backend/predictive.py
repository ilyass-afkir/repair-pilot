# Import Libraries
import os, time
import pandas as pd
import numpy as np
from backend.time_utils import *
from backend.os_utils import *
import plotly.express as px
import plotly.graph_objects as go 
from plotly.graph_objs._figure import Figure
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Functions

def demand_forecasting(df: pd.DataFrame, typ: str) -> pd.DataFrame:
    
    # Filter data by "typ"
    df_typ = df[df["typ"] == typ] 

    # Convert 'datum_wausgang' to datetime
    df_typ['datum_wausgang'] = df_typ['datum_wausgang'].apply(lambda x: date_time(x))

    # Extract year-month as a string
    df_typ['datum_wausgang_y_m'] = df_typ['datum_wausgang'].dt.strftime('%Y-%m')

    # Group by year-month and 'ARTIKEL', summing up 'MENGE'
    df_grouped = df_typ.groupby(['datum_wausgang_y_m', 'ARTIKEL'])['MENGE'].sum().reset_index()

    # Calculate total quantity per part
    part_total_quantity = df_grouped.groupby('ARTIKEL')['MENGE'].sum()

    # Calculate percentage contribution
    part_contribution = part_total_quantity / part_total_quantity.sum()

    # Select top 10% contributors
    top_10_percent_contributors = part_contribution[part_contribution >= part_contribution.quantile(0.9)].index.tolist()

    # Filter dataframe for top contributors
    df_filtered = df_grouped[df_grouped['ARTIKEL'].isin(top_10_percent_contributors)]

    # Create a pivot table to reshape the DataFrame
    df_reshaped = df_filtered.pivot_table(index='datum_wausgang_y_m', columns='ARTIKEL', values='MENGE', fill_value=0)

    # Reset the index to PeriodIndex and reindex to fill missing dates
    df_reshaped.index = pd.PeriodIndex(df_reshaped.index, freq='M')
    date_range = pd.period_range(start=df_reshaped.index.min(), end=df_reshaped.index.max(), freq='M')
    df_reshaped = df_reshaped.reindex(date_range, fill_value=0)

    # Reset the index and rename the columns if needed
    df_reshaped.reset_index(inplace=True)
    print(df_reshaped)

    # Cuda
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    print(df_reshaped.columns)

    sequence_length = 12  # Number of past months to consider
    num_features = len(df_reshaped.columns)-1 # Number of features

    # Convert the DataFrame to a numpy array
    data_array = df_reshaped.drop('datum_wausgang_y_m', axis=1).values

    # Prepare sequences and targets
    sequences = []
    targets = []
    for i in range(len(data_array) - sequence_length):
        sequences.append(data_array[i:i+sequence_length])
        targets.append(data_array[i+sequence_length])

    # Convert sequences and targets to numpy arrays
    sequences = np.array(sequences)
    targets = np.array(targets)

    split = int(0.8 * len(sequences))  # 80-20 split
    train_sequences = sequences[:split]
    train_targets = targets[:split]
    val_sequences = sequences[split:]
    val_targets = targets[split:]

    # Convert NumPy arrays to PyTorch tensors
    train_sequences_tensor = torch.tensor(train_sequences, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
    val_sequences_tensor = torch.tensor(val_sequences, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
    val_dataset = TensorDataset(val_sequences_tensor, val_targets_tensor)

    # Create dataloaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    

def predictive(typ: str, start_date: str, end_date: str) -> pd.DataFrame:
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("\033[1mPredictive analysis is now starting\033[0m")
    print("------------------------------------------------------------------------------------")
    
    # Start time measurement
    start_time = time.time()
    
    # Import preprocessed data
    print("Step 1: Importing preprocessed data")
    df_preprocessed = import_preprocessed_data(os.path.join(get_root_dir(), 'data', 'preprocessed'))
    
    print(len(df_preprocessed))
    print(len(df_preprocessed.columns))

    # Filter data by time
    print("Step 2: Filter data by time range")
    df_filtered = time_filter(df_preprocessed, start_date, end_date, "datum_wausgang")
    
    """
    # Time Analysis
    print("Step 3: Time analysis")   
    df_time_analysis = time_analysis(df_filtered, typ, start_date, end_date)
    fig_time_analysis = visualization_time_analysis(df_time_analysis, typ, start_date, end_date)
    save_df_as_csv(df_time_analysis, os.path.join(get_root_dir(), "reports", "time_analysis", "csv_files"), f"time_analysis_{typ}")
    save_figure(fig_time_analysis, os.path.join(get_root_dir(), "reports", "time_analysis", "figures"), typ, start_date, end_date)
    print("Step 4: Save Time analysis results (location: reports/time_analysis)")
     
    # Component Analysis
    print("Step 5: Component analysis")   
    df_component_analysis = component_anaylsis(df_filtered, typ, start_date, end_date)
    save_df_as_csv(df_component_analysis, os.path.join(get_root_dir(), "reports", "component_analysis"), f"component_analysis_{typ}")
    fig_components, fig_keyinfos = visualization_repair_analysis(df_component_analysis, typ, start_date, end_date)
    save_figure(fig_components, os.path.join(get_root_dir(), "reports", "repair_analysis", "figures", "barplot"), typ, start_date, end_date)
    save_figure(fig_keyinfos, os.path.join(get_root_dir(), "reports", "repair_analysis", "figures", "table"), typ, start_date, end_date)
    print("Step 6: Save Component analysis results (location: reports/component_analysis)")
    """
      # Clustering
    kmeans_clustering(df_filtered, typ, start_date, end_date)
    # Component dependencies (Data Mining: Association)
    df_components, rules = data_mining_component_associations(df_filtered, typ, min_support=0.5, metric="confidence", min_threshold = 0.5)
    fig = visualize_data_mining_component_associations(rules, typ, start_date, end_date)
    save_figure(fig, os.path.join(get_root_dir(), "reports", "component_dependencies", "figures"), typ, start_date, end_date)
    save_df_as_csv(rules, os.path.join(get_root_dir(), "reports", "component_analysis"), f"component_rules_{typ}")
    # Clustering
    kmeans_clustering(df_filtered, typ, start_date, end_date)
   
    # Demand Forecasting
    #df_reshaped = demand_forecasting(df_filtered, typ)


    # End time measurement
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Step 7: Data analysis is completed. Time taken: {time_taken} seconds")

