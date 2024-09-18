import os
from io import BytesIO
import base64
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import pydeck as pdk
import streamlit as st
from pysal.explore import esda
from pysal.lib import weights
from libpysal.weights import w_subset

# Import local functions from cflp_function (Assuming you have it correctly set up)
from Scripts.utils.cflp_function import *

# Constants
PADDING = 0
COLORMAP = 'magma'
VIEW_STATE = pdk.ViewState(longitude=4.390, latitude=51.891, zoom=8, bearing=0, pitch=0)

# Folder Path
CSV_FOLDER_PATH = './app_data/'  # Define your folder path here

# Helper function to list all CSV files in the folder
def list_csv_files(folder_path):
    """Returns a list of all CSV files in the specified folder."""
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Function to dynamically load all CSVs from the specified folder
def load_all_csvs(folder_path):
    """Loads all CSV files from the folder into a dictionary of DataFrames."""
    csv_files = list_csv_files(folder_path)
    dataframes = {}
    for file_path in csv_files:
        file_name = os.path.basename(file_path).replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                dataframes[file_name] = df
            else:
                raise ValueError(f"File {file_path} is empty.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"No data in file {file_path}.")
        except Exception as e:
            raise Exception(f"An error occurred while reading the file {file_path}: {str(e)}")
    return dataframes

# Loading all CSV dataframes
dataframes = load_all_csvs(CSV_FOLDER_PATH)

# Dynamically generated color mapping based on the loaded data
color_mapping = generate_color_mapping(COLORMAP)

# Setting page configuration
st.set_page_config(page_title="Geschiktheids Analyse", layout="wide")

# Setting markdown styling
st.markdown(
    """
    <style>
        div[data-testid="column"]:nth-of-type(4)
        {
            text-align: end;
        } 
    </style>
    """,
    unsafe_allow_html=True
)

#####

def load_gdf(gdf_path):
    """Function to load a GeoDataFrame from a file."""
    try:
        return gpd.read_file(gdf_path).set_index('hex9')
    except FileNotFoundError:
        raise FileNotFoundError(f"File {gdf_path} not found.")

# Loading spatial index data
idx = load_gdf('./app_data/h3_pzh_polygons.shp')

# Fuzzify input variables
@st.cache_data
def fuzzify(df, type="close", colormap_name=color_mapping):
    df_array = np.array(df['value'])
    fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / (df_array.max() - df_array.min())) if type == "close" else np.maximum(0, (df_array - df_array.min()) / (df_array.max() - df_array.min()))
    df['fuzzy'] = fuzzified_array.round(3)
    apply_color_mapping(df, 'fuzzy', color_mapping)
    return df

# Fuzzifying all loaded DataFrames dynamically
fuzzified_dataframes = {}
for key, df in dataframes.items():
    fuzzified_dataframes[key] = fuzzify(df, type='close' if 'close' in key else 'far')

# Preparing the arrays for analysis
all_arrays = {key: np.array(df['fuzzy']) for key, df in fuzzified_dataframes.items()}

#####

# Create empty layer
def create_empty_layer(df):
    empty_df = df[['hex9']]
    empty_df['color'] = '[0,0,0,0]'
    return empty_df

# Update layer based on selected variables
def update_layer(selected_variables, all_arrays, reference_df):
    if not selected_variables:
        return create_empty_layer(reference_df)

    selected_array_list = [all_arrays[key] for key in selected_variables]
    result_array = np.mean(selected_array_list, axis=0)
    hex_df = create_empty_layer(reference_df)
    hex_df['fuzzy'] = result_array
    apply_color_mapping(hex_df, 'fuzzy', color_mapping)
    hex_df['fuzzy'] = hex_df['fuzzy'].round(3)
    return hex_df

# Get potential sites based on suitability and spatial factors
def get_sites(df, w, g, idx, score_column: str = 'fuzzy', seed: int = 42):
    # Similar to your original implementation
    ...

# Generate pydeck visualization
@st.cache_resource
def generate_pydeck(df, view_state=VIEW_STATE):
    return pdk.Deck(
        initial_view_state=view_state,
        layers=[
            pdk.Layer(
                "H3HexagonLayer",
                df,
                pickable=True,
                stroked=True,
                filled=True,
                extruded=False,
                opacity=0.6,
                get_hexagon="hex9",
                get_fill_color='color', 
            ),
        ],
        tooltip={"text": "Geschiktheid: {fuzzy}"}
    )

# Create a variable legend for colormap
@st.cache_data
def generate_colormap_legend(label_left='Far', label_right='Near', cmap=plt.get_cmap(COLORMAP)):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.axis('off')
    ax.text(-10, 0.5, label_left, verticalalignment='center', horizontalalignment='right', fontsize=12)
    ax.text(266, 0.5, label_right, verticalalignment='center', horizontalalignment='left', fontsize=12)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    image_png = buffer.getvalue()
    plt.close(fig)
    image_base64 = base64.b64encode(image_png).decode()
    legend_html = f'''<div style="width: 100%; height: 300px; overflow: auto; padding: 10px;">
                      <img src="data:image/png;base64,{image_base64}" alt="Colorbar" 
                           style="max-width: 100%; max-height: 100%; height: auto; width: auto; 
                           display: block; margin-left: auto; margin-right: auto;">
                      </div>'''
    return legend_html

variable_legend_html = generate_colormap_legend(label_left='Minst Geschikt (0)', label_right='Meest Geschikt (1)',)

# Get layers for Pydeck visualization
@st.cache_data
def get_layers(hex_df):
    hex_fuzzy = pdk.Layer(
        "H3HexagonLayer",
        hex_df.reset_index(),
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        opacity=0.1,
        get_hexagon="hex9",
        get_fill_color='color', 
    )

    layers = [hex_fuzzy]
    return layers

# Plot suitability result
def plot_result(fig):
    if fig is not None:
        st.plotly_chart(fig, theme="streamlit")

#####

# Streamlit app starts here
def main(idx):
    initialize_session_state(idx)
    display_intro_text()
    plot_suitability_variables()
    perform_suitability_analysis()


# Initialize session state variables
def initialize_session_state(idx):
    if 'all_loi' not in st.session_state:
        st.session_state.all_loi = pd.DataFrame()
    if 'loi' not in st.session_state:
        st.session_state.loi = pd.DataFrame()
    if 'fig' not in st.session_state:
        st.session_state.fig = None
    if 'w' not in st.session_state:
        st.session_state.w = weights.Queen.from_dataframe(idx, use_index=True)
    if 'g' not in st.session_state:
        st.session_state.g = nx.read_graphml('./osm_network/extracts/G.graphml')

# Display introduction text
def display_intro_text():
    st.markdown("### Fase 1: Geschiktheidsanalyse - Potentiele locaties voor nieuwbouw projecten")
    st.markdown(
        "Bekijk de onderstaande kaarten, elk vertegenwoordigt een
