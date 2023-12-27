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
from sklearn.metrics import silhouette_score


# Functions


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





def diagnostic(typ: str, start_date: str, end_date: str) -> pd.DataFrame:
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("\033[1mDiagnostic analysis is now starting\033[0m")
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
    
    
      # Clustering
    kmeans_clustering(df_filtered, typ, start_date, end_date)
    # Component dependencies (Data Mining: Association)
    df_components, rules = data_mining_component_associations(df_filtered, typ, min_support=0.5, metric="confidence", min_threshold = 0.5)
    fig = visualize_data_mining_component_associations(rules, typ, start_date, end_date)
    save_figure(fig, os.path.join(get_root_dir(), "reports", "component_dependencies", "figures"), typ, start_date, end_date)
    save_df_as_csv(rules, os.path.join(get_root_dir(), "reports", "component_analysis"), f"component_rules_{typ}")
    # Clustering
    kmeans_clustering(df_filtered, typ, start_date, end_date)
   
    # End time measurement
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Step 7: Data analysis is completed. Time taken: {time_taken} seconds")

