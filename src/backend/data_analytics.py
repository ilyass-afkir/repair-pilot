# Import Libraries
import os, time
import pandas as pd
import numpy as np
from backend.time_utils import *
from backend.os_utils import *
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go 
from plotly.graph_objs._figure import Figure
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx

# Functions

def time_analysis(df: pd.DataFrame, typ: str, start_date: str, end_date: str) -> pd.DataFrame:

    """
    Calculate the average working days between each datum feature for the artikel of a product type in a given date range.

    Args:
        df (pandas.DataFrame): The input Dataframe.
        typ (str): The product type to filter by.
        start_date (str): The start date of the time range.
        end_date (str): The end date of the time range.
    
    Raises:
        ValueError: If "df" is empty.

    Returns:
        df_results (pd.DataFrame): A dataframe containing the results.
    
    Example:
        >>> df_time_analysis = time_analysis(df, "MM", "01.01.2017", "31.12.2018")
    
    """

    # Check if "df" is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. The time analysis can not be continued")
    
    # Create empty dataframe with columns to save the results
    df_results = pd.DataFrame(columns = ["Startdatum", "Enddatum", "Typ", 
                                        "Serviceaufträge",
                                        "Durchlaufzeit_waein_bis_wausgang",
                                        "repstart_bis_repend",
                                        "waein_bis_repstart",
                                        "repstart_bis_repkv",
                                        "repkv_bis_kv",
                                        "kv_bis_auftrag",
                                        "auftrag_bis_repend",
                                        "repend_bis_wausgang"])

    # Filter data by "typ"
    df_typ = df[df["typ"] == typ]
    
    # Drop duplicates based on "SERIVICEAUFTRAG"
    df_typ_serviceauftrag = df_typ.drop_duplicates(subset = "SERVICEAUFTRAG")
    
    # Set index
    i = 0
    
    # Save "start_date", "end_date", "typ", "Produktartikelnummer", "Produktartikelname" in df_results
    df_results.loc[i,"Startdatum"] = start_date
    df_results.loc[i,"Enddatum"] = end_date
    df_results.loc[i,"Typ"] = typ
    
    # Get and save total number of "SERVICEAUFTRAG" in df_results
    number_of_serviceaufträge = len(df_typ_serviceauftrag["SERVICEAUFTRAG"])
    df_results.loc[i,"Serviceaufträge"] = number_of_serviceaufträge
    
    # Average working days between "datum_waein" and "datum_wausgang" per Serviceauftrag
    df_results.loc[i,"Durchlaufzeit_waein_bis_wausgang"] = round(df_typ_serviceauftrag['Durchlaufzeit'].sum()/number_of_serviceaufträge, 2)
    
    # Average working days between "datum_repstart" and "datum_repend" per Serviceauftrag
    df_results.loc[i,"repstart_bis_repend"] = round(df_typ_serviceauftrag['repstart_bis_repend_arbeitstage'].sum()/number_of_serviceaufträge, 2)
    
    # Average working days between "datum_waein" and "datum_repstart" per Serviceauftrag
    df_results.loc[i,"waein_bis_repstart"] = round(df_typ_serviceauftrag['waein_bis_repstart_arbeitstage'].sum()/number_of_serviceaufträge, 2)
    
    # Average working days between "datum_repstart" and "datum_repkv" per Serviceauftrag
    df_results.loc[i,"repstart_bis_repkv"] = round(df_typ_serviceauftrag['repstart_bis_repkv_arbeitstage'].sum()/number_of_serviceaufträge, 2)
    
    # Average working days between "datum_repkv" and "datum_kv" per Serviceauftrag
    df_results.loc[i,"repkv_bis_kv"] = round(df_typ_serviceauftrag['repkv_bis_kv_arbeitstage'].sum()/number_of_serviceaufträge, 2)
    
    # Average working days between "datum_kv" and "datum_auftrag" per Serviceauftrag
    df_results.loc[i,"kv_bis_auftrag"] = round(df_typ_serviceauftrag['kv_bis_auftrag_arbeitstage'].sum()/number_of_serviceaufträge, 2)

    # Average working days between "datum_auftrag" and "datum_repend" per Serviceauftrag
    df_results.loc[i,"auftrag_bis_repend"] = round(df_typ_serviceauftrag['auftrag_bis_repend_arbeitstage'].sum()/number_of_serviceaufträge, 2)
    
    # Average working days  between "datum_repend" and "datum_wausgang" per Serviceauftrag
    df_results.loc[i,"repend_bis_wausgang"] = round(df_typ_serviceauftrag['repend_bis_wausgang_arbeitstage'].sum()/number_of_serviceaufträge, 2)

    return df_results


def visualization_time_analysis(df: pd.DataFrame, typ: str, start_date: str, end_date: str) -> Figure:
    
    """
    Visualize the time analysis results using a horizontal bar chart.

    Args:
        df (pandas.DataFrame): DataFrame containing the time analysis results.
        typ (str): The product type to filter by.
        start_date (str): The start date of the time range.
        end_date (str): The end date of the time range.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly Figure object representing the horizontal bar chart.

    Example:
        >>> fig = visualization_time_analysis(df_timeanalysis_results, "MM", 01.01.2018, "31.12.2018")

    """

    # Filter data by relevant columns and transpose the DataFrame to convert the column to a row
    df_plot = df.drop(["Startdatum", "Enddatum", "Typ", "Serviceaufträge", "repstart_bis_repend"], axis="columns").transpose().rename_axis("Arbeitsabschnitt").reset_index().rename(columns={0: "Arbeitstage_pro_Serviceauftrag"})

    # Get index of each datum feature in "Datum_Features" 
    index_dict = {row["Arbeitsabschnitt"]: index for index, row in df_plot.iterrows()}

    # Define starting points for the horizontal bar chart
    df_plot["Startpunkt"] = ""
    df_plot.loc[index_dict["Durchlaufzeit_waein_bis_wausgang"], "Startpunkt"] = 0
    df_plot.loc[index_dict["waein_bis_repstart"], "Startpunkt"] = 0
    df_plot.loc[index_dict["repstart_bis_repkv"], "Startpunkt"] = df_plot["Arbeitstage_pro_Serviceauftrag"].iloc[index_dict["waein_bis_repstart"]]
    df_plot.loc[index_dict["repkv_bis_kv"], "Startpunkt"] = df_plot["Arbeitstage_pro_Serviceauftrag"].iloc[index_dict["repstart_bis_repkv"]] + df_plot["Arbeitstage_pro_Serviceauftrag"].iloc[index_dict["waein_bis_repstart"]]
    df_plot.loc[index_dict["kv_bis_auftrag"], "Startpunkt"] = df_plot.loc[index_dict["repkv_bis_kv"], "Startpunkt"] + df_plot["Arbeitstage_pro_Serviceauftrag"].iloc[index_dict["repkv_bis_kv"]]
    df_plot.loc[index_dict["auftrag_bis_repend"], "Startpunkt"] = df_plot.loc[index_dict["kv_bis_auftrag"], "Startpunkt"] + df_plot["Arbeitstage_pro_Serviceauftrag"].iloc[index_dict["kv_bis_auftrag"]]
    df_plot.loc[index_dict["repend_bis_wausgang"], "Startpunkt"] = df_plot.loc[index_dict["auftrag_bis_repend"], "Startpunkt"] + df_plot["Arbeitstage_pro_Serviceauftrag"].iloc[index_dict["auftrag_bis_repend"]]
    
    # Create horizontal bar chart
    fig = px.bar(
        df_plot, 
        x="Arbeitstage_pro_Serviceauftrag", 
        y="Arbeitsabschnitt",
        text="Arbeitstage_pro_Serviceauftrag",
        orientation="h",
        width=1500,
        height=600,
        base="Startpunkt",
        color="Arbeitsabschnitt") 

    # Add number of working days to the horizontal bar chart
    fig.update_traces(textposition="outside")
        
    # Update layout
    fig.update_yaxes(title=None, showticklabels=False)
    fig.update_xaxes(title=None)
    fig.update_layout(title_x=0.5, title=f"Arbeitstage pro Serviceauftrag | Typ: {typ} | Zeitraum: {start_date}-{end_date}")
                        
    # Show the figure
    fig.show()

    return fig


def component_anaylsis(df: pd.DataFrame, typ: str, start_date: str, end_date: str) -> pd.DataFrame:

    """
    Analyze the components of a product type in a given date range.

    Args:
        df (pd.DataFrame): The input data 
        product_type (str): The product type to filter on.
        start_date (str): The start date of the analysis.
        end_date (str): The end date of the analysis.
    
    Raises:
        ValueError: If "df" is empty.

    Returns:
        df_results (pd.DataFrame): A DataFrame containing the results of the repair analysis.
    
    """

    # Check if "df" is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. The component analysis can not be continued")

    # Create empty dataframe with columns to save the results
    df_results = pd.DataFrame(columns = ["Startdatum", "Enddatum", "Typ", 
                                        "Serviceaufträge", 
                                        "Gesamte_Arbeitseinheiten", "Arbeitseinheiten_pro_Serviceauftrag", 
                                        "Serviceauftragzyklus", "Bauteilnummer", "Bauteilname", 
                                        "Gesamte_Bauteilmenge", "Bauteilmenge_pro_Serviceauftrag"])
    
    # Filter data by "typ"
    df_typ = df[df["typ"] == typ]

    # Loop through each "ARTIKEL" of a "typ"
    for (i, component) in enumerate(df_typ["ARTIKEL"].unique()):
        
        # Filter "df_type_artikel" by "ARTIKEL"
        df_typ_component = df_typ[df_typ["ARTIKEL"] == component]
    
        # Set the "start_date" and "end_date" 
        df_results.loc[i,"Startdatum"] = start_date
        df_results.loc[i,"Enddatum"] = end_date

        # Get the "Produkttyp", "Produktartikelnummer", "Produktartikelname", "Bauteilnummer" and "Bauteilname" 
        df_results.loc[i,"Typ"] = typ
        df_results.loc[i,"Bauteilnummer"] = component
        df_results.loc[i,"Bauteilname"] = df_typ_component["NAME"].iloc[0]

        # Get the total number of "SERVICEAUFTRAG" 
        number_of_serviceaufträge = len(df_typ["SERVICEAUFTRAG"].unique())
        df_results.loc[i,"Serviceaufträge"] = number_of_serviceaufträge

        # Calculate the total unit of work in the defined date range (start_date/end_date)
        df_results.loc[i,"Gesamte_Arbeitseinheiten"] = df_typ.drop_duplicates(subset = "SERVICEAUFTRAG")["Arbeitseinheit"].sum()
        
        # Calculate the average unit of work for a Serviceauftrag
        df_results.loc[i,"Arbeitseinheiten_pro_Serviceauftrag"] = round(df_typ.drop_duplicates(subset = "SERVICEAUFTRAG")["Arbeitseinheit"].sum()/number_of_serviceaufträge, 2)
        
        # Calculate the Serviceautrag cycle in the defined date range (start_date/end_date)
        df_results.loc[i,"Serviceauftragzyklus"] = round(calculate_working_days(start_date, end_date)/number_of_serviceaufträge, 2)

        # Calculate the total amount of parts used in the defined date range (start_date/end_date)
        df_results.loc[i,"Gesamte_Bauteilmenge"] = df_typ_component["MENGE"].sum()

        # Calculate the average amount of parts used in a Serviceauftrag
        df_results.loc[i,"Bauteilmenge_pro_Serviceauftrag"] = round(df_typ_component["MENGE"].sum()/number_of_serviceaufträge, 2)

    return df_results


def visualization_repair_analysis(df: pd.DataFrame, typ: str, start_date: str, end_date: str) -> Figure:
    
    
    # Plot for component amounts  
        
        # Sort df_results by "Gesamte_Bauteilmenge"
    df_plot_components = df.sort_values(by="Gesamte_Bauteilmenge", ascending=False)
        
        # Create horizontal bar chart 
    fig_components = px.bar(
        df_plot_components, 
        x="Bauteilnummer", 
        y="Gesamte_Bauteilmenge",
        text="Gesamte_Bauteilmenge",
        width=len(df_plot_components) * 30,
        height=len(df_plot_components) * 10)
    
        # Add number of working days to the horizontal bar chart
    fig_components.update_traces(textposition="outside")

        # Update layout
    fig_components.update_layout(title=f"Gesamte Bauteilmengen je Bauteilnummer | Typ: {typ} | Zeitraum: {start_date}-{end_date}")
    fig_components.update_yaxes(title=None) 
    fig_components.update_xaxes(title=None)
        
        # Show fig_components
    fig_components.show()

    # Plot repair key infos
    df_plot_keyinfos = df[["Serviceaufträge", "Gesamte_Arbeitseinheiten", "Arbeitseinheiten_pro_Serviceauftrag", "Serviceauftragzyklus"]].drop_duplicates(subset="Serviceaufträge").transpose().rename_axis("Feature").reset_index().rename(columns={0: "Ergebnis"})

        # Create a Plotly table
    fig_keyinfos = go.Figure(data=[go.Table(
        
        header=dict(
            values=[''],
            font_size=14,
            height=30,
            align='left'),
                    
        cells=dict(
            values=df_plot_keyinfos.T,  # Transpose the DataFrame to get values only
            font_size=14,
            height=30,
            align = "left"))])

        # Adjust the table layout
    fig_keyinfos.for_each_trace(lambda t: t.update(header_fill_color = 'rgba(0,0,0,0)'))
    fig_keyinfos.update_layout(title_x=0.5, width=700, height=350, title_text=f"Reparaturkennzahlen | Typ: {typ} | Zeitraum: {start_date}-{end_date}",)
    
        # Display the table 
    fig_keyinfos.show()

    return fig_components, fig_keyinfos
    

def data_mining_component_associations(df, typ, min_support: float, metric: str, min_threshold: float):
    
    # Filter data by "typ"
    df_typ = df[df["typ"] == typ]
    
    # Loop through each "SERVICEAUFTRAG" of a "typ" and save the "ARTIEKL" in a list of lists
    components_serviceauftrag = [df_typ[df_typ["SERVICEAUFTRAG"] == serviceauftrag]["ARTIKEL"].tolist() for serviceauftrag in df_typ["SERVICEAUFTRAG"].unique()]

    # Transform it into the right format TransactionEncoder 
    te = TransactionEncoder()
    te_ary = te.fit(components_serviceauftrag).transform(components_serviceauftrag)
    df_typ_components = pd.DataFrame(te_ary, columns=te.columns_)

    # Get frequent itemsets
    frequent_itemsets = apriori(df_typ_components, min_support=min_support, use_colnames=True)

    # Association rules
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    desired_items = [
        'K04886A',
        'K04172',
        'K04785',
        'K03832',
        'K03640'
    ]

    # Filter rules based on desired items in either antecedents or consequents within a string format
    rules = rules[
        (rules['antecedents'].apply(lambda x: any(item in x for item in desired_items))) &
        (rules['consequents'].apply(lambda x: any(item in x for item in desired_items)))
    ]

    print(rules)

    return frequent_itemsets, rules

def visualize_association_rules(rules):
    G = nx.DiGraph()

    for index, row in rules.iterrows():
        antecedents = '_und_'.join(row['antecedents'])
        consequents = '_und_'.join(row['consequents'])
        G.add_edge(antecedents, consequents, weight=row['confidence'])

    pos = nx.spring_layout(G)
    edge_labels = {(n1, n2): f"{G[n1][n2]['weight']:.2f}" for n1, n2 in G.edges}

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title('Association Rules Graph')
    plt.show()

def visualize_data_mining_component_associations(df: pd.DataFrame, typ: str, start_date: str, end_date: str):
    """
    Generate a Sankey diagram to visualize component associations based on data mining results.
    
    Args:
        df (pd.DataFrame): The input data 
        product_type (str): The product type to filter on.
        start_date (str): The start date of the analysis.
        end_date (str): The end date of the analysis.
    
    Returns:
    - fig (plotly.graph_objects.Figure): Sankey diagram displaying component associations
    
    Note:
    - The function expects the DataFrame (df) to contain 'antecedents', 'consequents', and 'confidence' columns.
    - It creates a Sankey diagram to represent associations between components.
    - The nodes and links represent antecedents and consequents, and their association strengths, respectively.
    """
    
    # Transform antecedents and consequents into labels
    label_antecedents = ['_und_'.join(item) for item in df["antecedents"].to_list()]
    label_consequents = ['_und_'.join(item) for item in df["consequents"].to_list()]
    
    # Map labels to numerical values for nodes in the Sankey diagram
    map_antecedents_to_number = {item: i for i, item in enumerate(list(set(label_antecedents)))}
    map_consequents_to_number = {item: len(map_antecedents_to_number.keys()) + i for i, item in enumerate(list(set(label_consequents)))}
    
    # Combine labels for antecedents and consequents
    label_antecedents_consequents = label_antecedents + label_consequents
    
    # Map labels to numerical values for links in the Sankey diagram
    source = [map_antecedents_to_number[item] for item in label_antecedents]
    target = [map_consequents_to_number[item] for item in label_consequents]
    value = [round(float(frozenset_item) * 100, 2) for frozenset_item in df["confidence"].to_list()]
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label_antecedents_consequents),
        link=dict(
            source=source,
            target=target,
            value=value))])
    
    # Update layout and display the figure
    fig.update_layout(title_x=0.5, title_text=f"Bauteilabhängigkeiten<br>Typ: {typ} | Prozentanteil: ≥50% | Zeitraum: {start_date}-{end_date}",
                      height=1000, width=800)
    fig.show()
    
    return fig


def kmeans_clustering(df: pd.DataFrame, typ: str, start_date: str, end_date: str) -> pd.DataFrame:

    # Create empty dataframe with columns to save the results
    df_results = pd.DataFrame(columns = ["Startdatum", "Enddatum", "Typ", 
                                        "Bauteilnummer", "Bauteilname", "Menge", "Frequenz", "Cluster"])

    # Filter data by "typ"
    df_typ = df[df["typ"] == typ] 
  
    # Loop through each "ARTIKEL" of a "typ"
    for part in df_typ["ARTIKEL"].unique():
        
        # Filter "df_typ" by "ARTIKEL"
        df_typ_part = df_typ[df_typ["ARTIKEL"] == part]
    
        # Calculate the part quantity
        part_quantity = df_typ_part["MENGE"].sum()

        # Calculate the part frequency
        part_frequency = len(df_typ_part["SERVICEAUFTRAG"].unique())

        # Save results
        df_results.loc[len(df_results)] = {
                                    "Startdatum": start_date,
                                    "Enddatum": end_date,
                                    "Typ": typ,
                                    "Bauteilnummer": part,
                                    "Bauteilname": df_typ_part.iloc[0]["NAME"],
                                    "Menge": part_quantity,
                                    "Frequenz": part_frequency,
                                    "Cluster": None}

    # Convert df_results to numpy array for clustering
    X = df_results[["Menge", "Frequenz"]].to_numpy()

    # Calculate the silhouette scores for different numbers of clusters
    silhouette_scores = []
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)

    # Find the optimal number of clusters
    max_score = max(silhouette_scores)
    optimal_num_clusters = silhouette_scores.index(max_score) + 2

    print("Optimal number of clusters: ", optimal_num_clusters)

    # Perform K-means clustering with optimal_num_clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters)
    cluster_labels = kmeans.fit_predict(X)  # Assuming X contains your feature data

    # Assign cluster labels to the df_results dataframe
    df_results["Cluster"] = cluster_labels
    print(df_results)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Find the top 5 critical Bauteilname based on Menge and Frequenz
    top_3_critical = df_results.nlargest(5, ['Menge', 'Frequenz'])

    # Filter df_results based on the top 3 critical Bauteilname
    top_3_results = df_results[df_results['Bauteilname'].isin(top_3_critical['Bauteilname'])]

    # Display the top 3 critical Bauteilname and their respective clusters along with Bauteilnummer
    critical_parts_list = top_3_results[['Bauteilnummer', 'Bauteilname', 'Cluster']]
    print(f"Top 3 critical parts:\n{critical_parts_list.to_string(index=False)}")
    # Plotting
    plt.figure(figsize=(8, 6))

    # Scatter plot for the clusters and their centers
    for i in range(len(cluster_centers)):
        cluster_data = df_results[df_results['Cluster'] == i]

        # Get the color for the cluster center
        cluster_color = plt.cm.tab10(i)  # Using tab10 colormap for distinct colors

        plt.scatter(cluster_data['Menge'], cluster_data['Frequenz'], label=f'Cluster {i}', color=cluster_color)
        # Plot cluster center only once in the legend
        if i == len(cluster_centers)-1:
            plt.scatter(cluster_centers[i][0], cluster_centers[i][1], marker='X', s=100, color="black", label='Cluster Center')
        else:
            plt.scatter(cluster_centers[i][0], cluster_centers[i][1], marker='X', s=100, color="black")

    translation_dict = {
    'Kohlebürste UX1; 2412': 'Carbon brush',
    'Rillenkugellager; 608 BRS': 'Deep groove ball bearing',
    'Dichtring; 4,5 x 7 x 1  U-Seal Standard': 'U-Seal Standard',
    'Öl; 80ml':'Oil 80ml',
    'Kabelklammer, Polyethylen; Größe 4':'Cable clamp, polyethylene'}

    line_styles = ['-', '--', '-.', ':', '.']

    line_style_counter = 0

    for index, row in df_results.iterrows():
        if row['Bauteilname'] in top_3_results['Bauteilname'].values:
            english_name = translation_dict.get(row['Bauteilname'], row['Bauteilname'])

            linestyle = line_styles[line_style_counter % len(line_styles)]

            line_style_counter += 1

            if linestyle == '-':
                marker = 'o'
            elif linestyle == '--':
                marker = 's'
            elif linestyle == '-.':
                marker = 'p'
            elif linestyle == ':':
                marker = 'D'
            else:
                marker = 'h'

            plt.scatter(row['Menge'], row['Frequenz'], marker=marker, s=150, edgecolor='black', facecolor='none', label=english_name)
    
    plt.xlabel('Quantity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

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
    
    



def data_analytics_main(typ: str, start_date: str, end_date: str) -> pd.DataFrame:
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("\033[1mData analysis is now starting\033[0m")
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

