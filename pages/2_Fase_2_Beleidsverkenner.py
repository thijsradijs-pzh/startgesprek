# pages/1_normalized_layers.py
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# Constants
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'magma'
VIEW_STATE = pdk.ViewState(longitude=4.390, latitude=51.891, zoom=8)

# Helper function to list and load CSVs
@st.cache_data
def list_and_load_csvs(folder_path):
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

def apply_color_mapping(df, value_column, colormap):
    """Applies a color map to a specified column of a DataFrame."""
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(colormap)
    df['color'] = df[value_column].apply(lambda x: [int(c * 255) for c in cmap(norm(x))[:3]])  # Get RGB values

# Fuzzify input variable
def fuzzify_layer(df, fuzz_type='close', colormap_name='magma'):
    """Fuzzifies the selected criterion and returns a fuzzified DataFrame."""
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

    return fuzzified_df

# Generate pydeck visualization
@st.cache_resource
def generate_pydeck(df_list, view_state=VIEW_STATE):
    """Generates Pydeck H3HexagonLayer visualization for each layer in the list."""
    layers = []

    for df in df_list:
        # Ensure 'hex9' is a column
        if 'hex9' not in df.columns:
            df = df.reset_index()

        hex_layer = pdk.Layer(
            "H3HexagonLayer",
            df,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            opacity=0.6,
            get_hexagon="hex9",
            get_fill_color="color",
        )
        layers.append(hex_layer)

    return pdk.Deck(
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "Fuzzy Value: {fuzzy}"}
    )

# Helper function to clean dataset names
def clean_dataset_name(name):
    """Replaces underscores with spaces for cleaner display."""
    return name.replace('_', ' ')

# Main function for the page
def main():
    st.title("Genormaliseerde Lagen Visualisatie")

    # Load data
    dataframes = list_and_load_csvs(CSV_FOLDER_PATH)

    # Create a dictionary to map clean names to original names
    clean_names_map = {clean_dataset_name(name): name for name in dataframes.keys()}

    # Sidebar selection
    with st.sidebar:
        st.header("Selecteer Criteria")
        selected_variables_clean = st.multiselect("Selecteer één of meerdere criteria om te visualiseren", list(clean_names_map.keys()))
        fuzz_type = st.selectbox("Selecteer of je juist dichtbij of ver van het criteria wilt zijn", ["Dichtbij", "Ver"])
        fuzz_type = 'close' if fuzz_type == "Dichtbij" else 'far'

    if selected_variables_clean:
        # Map clean names back to original names
        selected_variables = [clean_names_map[name] for name in selected_variables_clean]

        # Fuzzify and collect the dataframes
        fuzzified_dataframes = []
        for var in selected_variables:
            df = dataframes[var]
            fuzzified_df = fuzzify_layer(df, fuzz_type=fuzz_type, colormap_name=COLORMAP)
            fuzzified_dataframes.append(fuzzified_df)

        # Generate Pydeck visualization
        st.pydeck_chart(generate_pydeck(fuzzified_dataframes), use_container_width=True)

        # Export buttons for each dataset
        st.header("Exporteer Genormaliseerde Data")
        for i, df in enumerate(fuzzified_dataframes):
            csv = df[['hex9', 'fuzzy']].to_csv(index=False).encode('utf-8')
            dataset_name = selected_variables_clean[i]
            st.download_button(
                label=f"Download {dataset_name} als CSV",
                data=csv,
                file_name=f"{dataset_name}_fuzzified.csv",
                mime='text/csv'
            )

    else:
        st.info("Selecteer ten minste één criterium in de zijbalk om de genormaliseerde lagen te visualiseren.")

if __name__ == "__main__":
    main()
