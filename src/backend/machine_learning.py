# %% Import Libraries
    #Data Science
import pandas as pd
import numpy as np   
    # Machine Learning
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
    #Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
    #Other
from pathlib import Path
import os
from backend.os_utils import *
from backend.time_utils import *


# Functions
# Goal. Cluster parts of a product based on 2 key parameters: total number of used parts and frequency (meanin occurances in repair orders)
def kmeans_clustering(df: pd.DataFrame, product_type: str, start_date: str, end_date: str) -> pd.DataFrame:

    # Create empty dataframe with columns to save the results
    df_results = pd.DataFrame(columns = ["Startdatum", "Enddatum", "Typ", "Artikelnummer", "Artikelname", 
                                        "Bauteilnummer", "Bauteilname", "Cluster"])

    # Create dataframe that will be used for Clustering
    df_clustering = pd.DataFrame(columns = ["ARTIKEL", "MENGE"])

    # Filter data by "producttyp"
    df_typ = df[df["typ"] == product_type]
    
    # Initialize row indexes
    #i = 0

    # Loop through each "artikel" of a "typ"
    for artikel in df_typ["artikel"].unique():

        # Filter "df_type" by "artikel"
        df_typ_artikel = df_typ[df_typ["artikel"] == artikel]
        
        # Loop through each "ARTIKEL" of a "artikel"
        for i, component in enumerate(df_typ_artikel["ARTIKEL"].unique()):
          
            # Filter "df_type_artikel" by "ARTIKEL"
            df_typ_artikel_component = df_typ_artikel[df_typ_artikel["ARTIKEL"] == component]
        
            # Calculate total number of components in the defined time range
            df_clustering.loc[i, "MENGE"] = df_typ_artikel_component["MENGE"].sum()

            # Add the component
            df_clustering.loc[i, "ARTIKEL"] = component

          
        
        # Convert df_clustering to numpy array
        X = df_clustering["MENGE"].to_numpy().reshape(-1,1)

        # Calculate the silhouette scores for different numbers of clusters
        silhouette_scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(X)
            score = silhouette_score(X, cluster_labels)
            silhouette_scores.append(score)

        # Find the optimal number of clusters
        max_score = max(silhouette_scores)
        optimal_n_clusters = silhouette_scores.index(max_score) + 2

        print("Optimal number of clusters: ", optimal_n_clusters)

        # Perform K-means clustering with k = n_clusters
        kmeans = KMeans(n_clusters = optimal_n_clusters)
        kmeans.fit(X)

        # Save results in df_results
        labels = kmeans.labels_
        for j, label in enumerate(labels):
            indices = df_typ_artikel.index[df_typ_artikel['ARTIKEL'] == df_clustering.loc[j, "ARTIKEL"]].tolist()
            row = {
                "Startdatum": start_date,
                "Enddatum": end_date,
                "Typ": product_type,
                "Artikelnummer": artikel,
                "Artikelname": df_typ_artikel['artname'].iloc[0],
                "Bauteilnummer": df_clustering.loc[j, "ARTIKEL"],
                "Bauteilname": df_typ_artikel["NAME"].iloc[indices[0]],
                "Cluster": label}

            globals()[f"df_{artikel}"] = df_results.append(row, ignore_index=True)

        
        
        return df_results


def kmeans_clustering_main(product_type, datum_feature, start_date, end_date):
    
    # Import preprocessed data
    df_preprocessed = import_preprocessed_data(os.path.join(get_parent_dir(), 'data', 'preprocessed'))
    print("Import preprocessed data status: completed")

    # Filter data
    df_filtered = time_filter(df_preprocessed, datum_feature, start_date, end_date)
    print("Time filter status: completed")

    # KMeans Clustering
    df_clustering = kmeans_clustering(df_filtered, product_type, start_date, end_date)

    return df_clustering
    
# Main Program
if __name__ == '__main__':
    df_clustering = kmeans_clustering_main("MM", "datum_wausgang", "01.01.2017", "31.12.2021")



# %%
