import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go 
from plotly.graph_objs._figure import Figure
from utils.const import FEATURED_FILE_PATH, RESULTS_PATH
from utils.data import import_csv_as_df, save_df_as_csv
from utils.time import time_filter 
from utils.fig import save_plotly_as_html


def time_analysis(df: pd.DataFrame, typ: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Calculate the average working days between each datum feature for the artikel of a product type in a given date range.

    Args:
        df (pandas.DataFrame): The input Dataframe.
        typ (str): The product type to filter by.
        start_date (str): The start date of the time range.
        end_date (str): The end date of the time range.
    
    Returns:
        df_results (pd.DataFrame): A dataframe containing the results.
    """
    # Create empty dataframe with columns to save the results
    df_results = pd.DataFrame(columns = ["Startdatum", "Enddatum", "Typ", 
                                        "Serviceaufträge",
                                        "Durchlaufzeit",
                                        "waein_bis_repstart_arb",
                                        "repstart_bis_repkv_arb",
                                        "repkv_bis_kv_arb",
                                        "kv_bis_auftrag_arb",
                                        "auftrag_bis_repend_arb",
                                        "repend_bis_wausgang_arb"])

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
    
    column_names = ['Durchlaufzeit_arb', 'waein_bis_repstart_arb',
                'repstart_bis_repkv_arb', 'repkv_bis_kv_arb', 'kv_bis_auftrag_arb',
                'auftrag_bis_repend_arb', 'repend_bis_wausgang_arb']

    for col_name in column_names:
        avg_col_name = f"{col_name}_avg"
        df_results.loc[i, avg_col_name] = round(df_typ_serviceauftrag[col_name].sum() / number_of_serviceaufträge, 2)
        
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
    # Ensure the column names match those produced in the time_analysis function
    relevant_columns = [
        "Durchlaufzeit_arb_avg",
        "waein_bis_repstart_arb_avg",
        "repstart_bis_repkv_arb_avg",
        "repkv_bis_kv_arb_avg",
        "kv_bis_auftrag_arb_avg",
        "auftrag_bis_repend_arb_avg",
        "repend_bis_wausgang_arb_avg"
    ]

    # Filter and prepare data for plotting
    df_plot = df[relevant_columns].transpose().reset_index()
    df_plot.columns = ['Arbeitsabschnitt', 'Arbeitstage']

    # Calculate cumulative sum for the starting point of each bar segment
    df_plot['Startpunkt'] = df_plot['Arbeitstage'].cumsum() - df_plot['Arbeitstage']

    # Create a horizontal bar chart
    fig = px.bar(
        df_plot,
        x='Arbeitstage',
        y='Arbeitsabschnitt',
        orientation='h',
        text='Arbeitstage',
        base='Startpunkt',
        title=f"Arbeitstage pro Serviceauftrag | Typ: {typ} | Zeitraum: {start_date} - {end_date}",
        labels={'Arbeitstage':'Arbeitstage pro Serviceauftrag', 'Arbeitsabschnitt':'Arbeitsabschnitt'}
    )

    # Update traces for better text visibility and style
    fig.update_traces(textposition='inside')

    # Update layout for a cleaner look
    fig.update_layout(
        xaxis_title="Arbeitstage",
        yaxis_title="Arbeitsabschnitt",
        height=600,
        width=800
    )

    fig.show()

    return fig


def process_anaylsis(df: pd.DataFrame, typ: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Analyze the repair process of a product type in a given date range.

    Args:
        df (pd.DataFrame): The input data 
        product_type (str): The product type to filter on.
        start_date (str): The start date of the analysis.
        end_date (str): The end date of the analysis.
    
    Returns:
        df_results (pd.DataFrame): A DataFrame containing the results of the repair analysis.
    """
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


def visualization_process_analysis(df: pd.DataFrame, typ: str, start_date: str, end_date: str) -> Figure:
    
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
    
    df_preprocessed = import_csv_as_df(FEATURED_FILE_PATH)
    df_filtered = time_filter(df_preprocessed, start_date, end_date, "datum_wausgang")
    df_time = time_analysis(df_filtered, typ, start_date, end_date)
    fig_time = visualization_time_analysis(df_time, typ, start_date, end_date)
    save_df_as_csv(df_time, Path.joinpath(RESULTS_PATH,'time_analysis','time_periods.csv'))
    save_plotly_as_html(fig_time, Path.joinpath(RESULTS_PATH,'time_analysis','time_periods.html'))
    
    df_repair_process = process_anaylsis(df_preprocessed, start_date, end_date, "datum_wausgang")
    fig_process = visualization_process_analysis(df_preprocessed, start_date, end_date, "datum_wausgang")
    save_df_as_csv(df_repair_process, Path.joinpath(RESULTS_PATH,'process_analysis','process.csv'))
    save_plotly_as_html(fig_process, Path.joinpath(RESULTS_PATH,'process_analysis','process.html'))
     
    

