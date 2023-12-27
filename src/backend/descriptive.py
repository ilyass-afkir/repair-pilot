# Import Libraries
import os, time
import pandas as pd
import numpy as np
from backend.time_utils import *
from backend.os_utils import *
import plotly.express as px
import plotly.graph_objects as go 
from plotly.graph_objs._figure import Figure


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
    

def descriptive(typ: str, start_date: str, end_date: str) -> pd.DataFrame:
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("\033[1mDescriptive analysis is now starting\033[0m")
    print("------------------------------------------------------------------------------------")
    
    # Start time measurement
    start_time = time.time()
    
    # Import preprocessed data
    print("Step 1: Importing preprocessed data")
    df_preprocessed = import_preprocessed_data(os.path.join(get_root_dir(), 'data', 'preprocessed'))
    
    # Filter data by time
    print("Step 2: Filter data by time range")
    df_filtered = time_filter(df_preprocessed, start_date, end_date, "datum_wausgang")
    
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
    
    # End time measurement
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Step 7: Data analysis is completed. Time taken: {time_taken} seconds")

