import os
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import networkx as nx
from pysal.explore import esda
from pysal.lib import weights
from libpysal.weights import w_subset
from io import BytesIO
import base64

# Constants
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'magma'
VIEW_STATE = pdk.ViewState(longitude=4.390, latitude=51.891, zoom=8)

# Helper function to list all CSV files in the folder
@st.cache_data
def list_and_load_csvs(folder_path):
    """Lists and loads all CSV files from the specified folder."""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df_name = os.path.splitext(file)[0]
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                dataframes[df_name] = df
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return dataframes

# Load GeoDataFrame from shapefile
@st.cache_data
def load_gdf(gdf_path):
    """Loads a GeoDataFrame from a shapefile."""
    return gpd.read_file(gdf_path).set_index('hex9')

def apply_color_mapping(df, value_column, colormap):
    """Applies a color map to a specified column of a DataFrame."""
    df['color'] = df[value_column].apply(lambda x: [int(c * 255) for c in colormap(x)[:3]])  # Get RGB values

# Fuzzify input variables with "close" and "far", returning each fuzzified layer individually
def fuzzify_each_layer(df_list, fuzz_type='close', colormap_name='magma'):
    """Fuzzifies each selected criterion separately and returns a list of fuzzified DataFrames."""
    fuzzified_dataframes = []
    
    for df in df_list:
        df_array = np.array(df['value'])
        
        # Apply fuzzification depending on the fuzz_type
        if fuzz_type == "close":
            fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / (df_array.max() - df_array.min()))
        else:  # fuzz_type == "far"
            fuzzified_array = np.maximum(0, (df_array - df_array.min()) / (df_array.max() - df_array.min()))
        
        # Create a new DataFrame for the fuzzified result
        fuzzified_df = df.copy()  # Start with a copy of the original DataFrame
        fuzzified_df['fuzzy'] = np.round(fuzzified_array, 3)  # Add the fuzzified values
        
        # Apply the colormap
        colormap = plt.get_cmap(colormap_name)
        apply_color_mapping(fuzzified_df, 'fuzzy', colormap)
        
        # Append fuzzified dataframe to the list
        fuzzified_dataframes.append(fuzzified_df)
    
    return fuzzified_dataframes

# Stack the individual fuzzified layers into a single DataFrame
def stack_fuzzified_layers(fuzzified_dataframes):
    """Stacks multiple fuzzified DataFrames by joining them on 'hex9' index."""
    # Start with the first DataFrame in the list
    stacked_df = fuzzified_dataframes[0][['hex9', 'fuzzy']].copy()
    stacked_df.rename(columns={'fuzzy': 'fuzzy_1'}, inplace=True)  # Rename the fuzzy column to fuzzy_1

    # Add remaining fuzzified DataFrames
    for i, df in enumerate(fuzzified_dataframes[1:], start=2):
        stacked_df = stacked_df.join(df[['hex9', 'fuzzy']].set_index('hex9'), on='hex9', rsuffix=f'_{i}')
    
    return stacked_df

# Spatial suitability analysis on the stacked DataFrame
def perform_spatial_analysis_on_stack(stacked_df, idx, w, g, seed=42):
    """Performs spatial suitability analysis on the stacked DataFrame with multiple fuzzified layers."""
    # Drop rows with NaN values and ensure alignment with the spatial index
    stacked_df = stacked_df.dropna().set_index('hex9')
    stacked_df = stacked_df.loc[stacked_df.index.intersection(idx.index)]
    
    # Apply spatial weights and perform local Moran's I analysis on the stacked data
    w_subset_result = w_subset(w, stacked_df.index)
    
    # Sum the fuzzified columns to get an overall score for each hexagon
    stacked_df['fuzzy_sum'] = stacked_df.filter(like='fuzzy').sum(axis=1)
    
    # Perform Moran's I spatial autocorrelation analysis on the summed fuzzy values
    lisa = esda.Moran_Local(stacked_df['fuzzy_sum'], w_subset_result, seed=seed)
    significant_locations = stacked_df[(lisa.q == 1) & (lisa.p_sim < 0.01)].index
    
    H = g.subgraph(significant_locations)
    H_undirected = nx.Graph(H.to_undirected())
    
    return stacked_df.loc[significant_locations]

# Generate pydeck visualization - Modified to visualize selected hexagons and full layer
@st.cache_resource
def generate_pydeck(df=None, selected_hexagons=None, view_state=VIEW_STATE):
    """Generates Pydeck H3HexagonLayer visualization with both full dataset and selected hexagons."""
    layers = []

    # Add a layer for all hexagons
    if df is not None:
        full_hex_layer = pdk.Layer(
            "H3HexagonLayer",
            df,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            opacity=0.6,
            get_hexagon="hex9",
            get_fill_color="color",  # Use the color field generated by apply_color_mapping
        )
        layers.append(full_hex_layer)

    # Add a layer for selected hexagons, highlighting them in red
    if selected_hexagons is not None:
        selected_hex_layer = pdk.Layer(
            "H3HexagonLayer",
            selected_hexagons,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            opacity=0.9,
            get_hexagon="hex9",
            get_fill_color=[255, 0, 0],  # Red color for selected hexagons
        )
        layers.append(selected_hex_layer)

    return pdk.Deck(
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "Geschikt"}
    )

# Helper function to clean dataset names
def clean_dataset_name(name):
    """Replaces underscores with spaces for cleaner display."""
    return name.replace('_', ' ')

# Perform suitability analysis and update layers for stacked fuzzified layers
def perform_suitability_analysis_on_stack(dataframes, idx):
    # Create a dictionary to map clean names to original names
    clean_names_map = {clean_dataset_name(name): name for name in dataframes.keys()}
    
    with st.sidebar.form("suitability_analysis_form"):
        # Use clean names in the multiselects
        selected_variables_close_clean = st.multiselect(":one: Selecteer criteria waar je in de buurt wilt zitten", list(clean_names_map.keys()), key='close')
        selected_variables_far_clean = st.multiselect(":two: Selecteer criteria waar je verder vanaf wilt zitten", list(clean_names_map.keys()), key='far')
        submit_button = st.form_submit_button("Bouw Geschiktheidskaart")
    
    if submit_button and (selected_variables_close_clean or selected_variables_far_clean):
        # Map clean names back to original names
        selected_variables_close = [clean_names_map[name] for name in selected_variables_close_clean]
        selected_variables_far = [clean_names_map[name] for name in selected_variables_far_clean]
        
        # Perform fuzzify on the selected datasets for "close" and "far" separately
        fuzzified_dataframes_close = fuzzify_each_layer([dataframes[var] for var in selected_variables_close], 'close', COLORMAP)
        fuzzified_dataframes_far = fuzzify_each_layer([dataframes[var] for var in selected_variables_far], 'far', COLORMAP)

        # Stack fuzzified layers
        stacked_df_close = stack_fuzzified_layers(fuzzified_dataframes_close) if fuzzified_dataframes_close else None
        stacked_df_far = stack_fuzzified_layers(fuzzified_dataframes_far) if fuzzified_dataframes_far else None

        # Combine the stacked DataFrames for 'close' and 'far' criteria
        if stacked_df_close is not None and stacked_df_far is not None:
            stacked_df = stacked_df_close.join(stacked_df_far.set_index('hex9'), on='hex9', rsuffix='_far')
        elif stacked_df_close is not None:
            stacked_df = stacked_df_close
        else:
            stacked_df = stacked_df_far

        # Spatial analysis on the stacked layers
        try:
            all_loi = perform_spatial_analysis_on_stack(stacked_df, idx, st.session_state.w, st.session_state.g)
            if not all_loi.empty:
                st.session_state.all_loi = all_loi
                st.write(f"Aantal Potentiële Locaties: {len(all_loi)}")
    
                # Reset indices
                all_loi_reset = all_loi.reset_index()
                stacked_df_reset = stacked_df.reset_index()
    
                # Create Pydeck chart
                st.pydeck_chart(
                generate_pydeck(
                df=stacked_df_reset, 
                selected_hexagons=all_loi_reset
                ), 
                    use_container_width=True
                )
    
                # Plot distribution
                fig = ff.create_distplot(
                    [all_loi['fuzzy_sum'].tolist()], 
                    ['Distribution of Fuzzy Sum'], 
                    show_hist=False, 
                    bin_size=0.02
                )
                st.plotly_chart(fig, use_container_width=True)
            # if not all_loi.empty:
            #     st.session_state.all_loi = all_loi
            #     st.write(f"Aantal Potentiële Locaties: {len(all_loi)}")
                
            #     # Create Pydeck chart with both the full dataset and selected hexagons visualized
            #     st.pydeck_chart(generate_pydeck(df=stacked_df, selected_hexagons=all_loi), use_container_width=True)
                
            #     # Plot distribution of the fuzzy_sum score
            #     fig = ff.create_distplot([all_loi['fuzzy_sum'].tolist()], ['Distribution of Fuzzy Sum'], show_hist=False, bin_size=0.02)
            #     st.plotly_chart(fig, use_container_width=True)
        except ValueError as e:
            st.error(str(e))

# Session state initialization
def initialize_session_state(idx):
    if 'w' not in st.session_state:
        st.session_state.w = weights.Queen.from_dataframe(idx, use_index=True)
    if 'g' not in st.session_state:
        st.session_state.g = nx.read_graphml('./app_data/G.graphml')

# Display introductory text
def display_intro_text():
    st.markdown("### Fase 1: Geschiktheidsanalyse")
    st.markdown(
        "Gebruik de dropdown om verschillende criteria te visualiseren en te analyseren.",
        unsafe_allow_html=True
    )

# Display dropdown for CSV selection and visualize the selected CSV
def display_csv_selection(dataframes):
    # Create a dictionary to map clean names to original names
    clean_names_map = {clean_dataset_name(name): name for name in dataframes.keys()}
    
    # Use clean names in the dropdown
    selected_variable_clean = st.selectbox("Selecteer dataset", list(clean_names_map.keys()))
    
    # Use the original dataset name to update the layer
    if selected_variable_clean:
        selected_variable = clean_names_map[selected_variable_clean]
        updated_layer = update_layer(selected_variable, dataframes)
        st.pydeck_chart(generate_pydeck(updated_layer), use_container_width=True)

# Update layer based on selected variables (non-fuzzified)
def update_layer(selected_variable, all_dataframes):
    """Updates the visualization layer with selected variable."""
    df = all_dataframes[selected_variable]
    hex_df = create_empty_layer(df)
    hex_df['value'] = df['value']
    # Apply colormap
    color_map = plt.get_cmap('magma')
    apply_color_mapping(hex_df, 'value', color_map)
    return hex_df

# Prepare empty layer
def create_empty_layer(df):
    """Creates an empty layer with transparent color."""
    empty_df = df[['hex9']].copy()
    empty_df['color'] = '[0,0,0,0]'
    return empty_df

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="Geschiktheids Analyse", layout="wide")

    # Load data
    dataframes = list_and_load_csvs(CSV_FOLDER_PATH)
    idx = load_gdf('./app_data/h3_pzh_polygons.shp')
    
    # Initialize session state
    initialize_session_state(idx)

    # Display UI
    display_intro_text()
    display_csv_selection(dataframes)
    perform_suitability_analysis_on_stack(dataframes, idx)

# Run the Streamlit app
if __name__ == "__main__":
    main()
