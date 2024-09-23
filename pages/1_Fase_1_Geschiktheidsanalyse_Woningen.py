import os
import base64
from io import BytesIO
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import networkx as nx
from pysal.explore import esda
from pysal.lib import weights
from libpysal.weights import w_subset

# Import local functions from cflp_function (Ensure this path is correct)
from Scripts.utils.cflp_function import *

# Constants
CSV_FOLDER_PATH = './app_data/'
PADDING = 0
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
    """
    Applies a color map to a specified column of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame with the column to color map.
        value_column (str): The column name containing normalized values [0, 1].
        colormap (function): A callable colormap function (e.g., plt.get_cmap()).
    """
    # Apply the colormap to the normalized values in the value_column
    df['color'] = df[value_column].apply(lambda x: [int(c * 255) for c in colormap(x)[:3]])  # Get RGB values

# Fuzzify input variables
@st.cache_data
def fuzzify(df, colormap_name='magma'):
    """
    Normalizes 'value' column to [0,1] and applies color mapping.
    """
    df['fuzzy'] = np.clip((df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min()), 0, 1).round(3)
    
    # Use a colormap function instead of a string
    colormap = plt.get_cmap(colormap_name)
    
    # Apply the colormap to the 'fuzzy' values
    apply_color_mapping(df, 'fuzzy', colormap)
    return df


# Prepare empty layer
def create_empty_layer(df):
    """Creates an empty layer with transparent color."""
    empty_df = df[['hex9']].copy()
    empty_df['color'] = '[0,0,0,0]'
    return empty_df

# Update layer based on selected variables
def update_layer(selected_variables, all_arrays, reference_df):
    if not selected_variables:
        return create_empty_layer(reference_df)
    result_array = np.mean([all_arrays[var] for var in selected_variables], axis=0)
    hex_df = create_empty_layer(reference_df)
    hex_df['fuzzy'] = result_array.round(3)
    # Use a colormap function instead of a string
    color_map = plt.get_cmap('magma')
    apply_color_mapping(hex_df, 'fuzzy', color_map)
    return hex_df

# Spatial suitability analysis
def get_sites(df, w, g, idx, score_column='fuzzy', seed=42):
    """Analyzes potential sites for new construction based on suitability scores and spatial analysis."""
    df = df.dropna(subset=[score_column]).drop_duplicates(subset='hex9').set_index('hex9')
    df = df.loc[df.index.intersection(idx.index)]
    w_subset_result = w_subset(w, df.index)
    
    lisa = esda.Moran_Local(df[score_column], w_subset_result, seed=seed)
    significant_locations = df[(lisa.q == 1) & (lisa.p_sim < 0.01)].index
    
    H = g.subgraph(significant_locations)
    H_undirected = nx.Graph(H.to_undirected())
    
    return df.loc[significant_locations]

# Generate pydeck visualization
@st.cache_resource
def generate_pydeck(df, view_state=VIEW_STATE):
    """Generates Pydeck H3HexagonLayer visualization."""
    return pdk.Deck(
        initial_view_state=view_state,
        layers=[pdk.Layer(
            "H3HexagonLayer",
            df,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            opacity=0.6,
            get_hexagon="hex9",
            get_fill_color='color',
        )],
        tooltip={"text": "Geschiktheid: {fuzzy}"}
    )

# Generate colormap legend
@st.cache_data
def generate_colormap_legend(label_left='Minst Geschikt (0)', label_right='Meest Geschikt (1)', cmap=plt.get_cmap(COLORMAP)):
    """Generates a base64-encoded color map legend."""
    gradient = np.vstack((np.linspace(0, 1, 256),) * 2)
    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.axis('off')
    ax.text(-10, 0.5, label_left, verticalalignment='center', horizontalalignment='right', fontsize=12)
    ax.text(266, 0.5, label_right, verticalalignment='center', horizontalalignment='left', fontsize=12)
    
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return f'<div><img src="data:image/png;base64,{img_base64}" alt="Colorbar" style="display:block;margin:auto;"></div>'

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="Geschiktheids Analyse", layout="wide")

    # Load data
    dataframes = list_and_load_csvs(CSV_FOLDER_PATH)
    idx = load_gdf('./app_data/h3_pzh_polygons.shp')
    
    # Fuzzify data
    fuzzified_dataframes = {key: fuzzify(df, COLORMAP) for key, df in dataframes.items()}
    all_arrays = {key: np.array(df['fuzzy']) for key, df in fuzzified_dataframes.items()}
    
    # Initialize session state
    initialize_session_state(idx)

    # Display UI
    display_intro_text()
    plot_suitability_variables(fuzzified_dataframes)
    perform_suitability_analysis(all_arrays, fuzzified_dataframes, dataframes, idx)

# Session state initialization
def initialize_session_state(idx):
    if 'w' not in st.session_state:
        st.session_state.w = weights.Queen.from_dataframe(idx, use_index=True)
    if 'g' not in st.session_state:
        st.session_state.g = nx.read_graphml('./app_data/G.graphml')

# Display introductory text
def display_intro_text():
    st.markdown("### Fase 1: Geschiktheidsanalyse - Potentiele locaties voor nieuwbouw projecten")
    st.markdown(
        "Deze kaart toont verschillende criteria die belangrijk zijn voor het bepalen van de geschiktheid van gebieden voor nieuwbouw.",
        unsafe_allow_html=True
    )

# Plot suitability variables
def plot_suitability_variables(fuzzified_dataframes):
    cols = st.columns(3)
    for i, (key, df) in enumerate(fuzzified_dataframes.items()):
        column = cols[i % 3]
        column.markdown(f"**{key.replace('_', ' ').title()}**")
        column.pydeck_chart(generate_pydeck(df), use_container_width=True)
    
    st.markdown(generate_colormap_legend(), unsafe_allow_html=True)

# Perform suitability analysis and update layer
def perform_suitability_analysis(all_arrays, fuzzified_dataframes, dataframes, idx):
    with st.sidebar.form("suitability_analysis_form"):
        selected_variables = st.multiselect(":one: Selecteer Criteria", list(all_arrays.keys()))
        submit_button = st.form_submit_button("Bouw Geschiktheidskaart")

    if submit_button and selected_variables:
        hex_df = update_layer(selected_variables, all_arrays, dataframes[next(iter(dataframes))])

        # Spatial analysis and display
        try:
            all_loi = get_sites(hex_df, st.session_state.w, st.session_state.g, idx)
            if not all_loi.empty:
                st.session_state.all_loi = all_loi
                st.write(f"Aantal Potentiële Locaties: {len(all_loi)}")
                fig = ff.create_distplot([all_loi['fuzzy'].tolist()], ['Distribution'], show_hist=False, bin_size=0.02)
                st.plotly_chart(fig, use_container_width=True)
        except ValueError as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == "__main__":
    main()
