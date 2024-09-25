import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import streamlit as st
import geopandas as gpd

# Constants
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'magma'
VIEW_STATE = pdk.ViewState(longitude=4.390, latitude=51.891, zoom=8)

# Load the complete PZH H3 hexagon layer (all possible hexagons)
@st.cache_data
def load_complete_hexagons(gdf_path):
    """Loads all H3 hexagons from the shapefile for the PZH region."""
    gdf = gpd.read_file(gdf_path)
    if 'hex9' not in gdf.columns:
        st.error("The shapefile must contain a 'hex9' column.")
        return None
    return gdf.set_index('hex9')

# Helper function to list and load CSVs, ensuring reindexing with complete hexagons
@st.cache_data
def list_and_load_csvs(folder_path, all_hexagons):
    """Lists and loads all CSV files from the specified folder."""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df_name = os.path.splitext(file)[0]
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                if 'hex9' not in df.columns:
                    st.error(f"'hex9' column not found in {file}")
                    continue
                df = df.set_index('hex9')
                # Reindex to include all hexagons, fill missing values with zero
                df = df.reindex(all_hexagons, fill_value=0)
                df.reset_index(inplace=True)  # Ensure 'hex9' is a column again
                dataframes[df_name] = df
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return dataframes

# Fuzzify each input layer
def fuzzify_layer(df, fuzz_type='close', colormap_name='magma'):
    """Fuzzifies the selected criterion and returns a fuzzified DataFrame."""
    df_array = np.array(df['value'])

    # Avoid division by zero
    range_diff = df_array.max() - df_array.min()
    if range_diff == 0:
        range_diff = 1

    # Apply fuzzification logic based on fuzz_type
    if fuzz_type == 'close':
        fuzzified_array = np.maximum(0, (df_array - df_array.min()) / range_diff)
    else:  # 'far'
        fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / range_diff)

    # Create a fuzzified DataFrame
    fuzzified_df = df.copy()
    fuzzified_df['fuzzy'] = np.round(fuzzified_array, 3)  # Rounded to 3 decimals

    # Apply the colormap
    colormap = plt.get_cmap(colormap_name)
    norm = plt.Normalize(vmin=0, vmax=1)
    fuzzified_df['color'] = fuzzified_df['fuzzy'].apply(lambda x: [int(c * 255) for c in colormap(norm(x))[:3]])

    return fuzzified_df

# Generate pydeck visualization
def generate_pydeck(df_list, view_state=VIEW_STATE):
    """Generates Pydeck H3HexagonLayer visualization for each layer."""
    layers = []
    
    for df in df_list:
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

# Main function
def main():
    st.title("Analyseer de genormaliseerde lagen voor de juiste keuze voor de geschiktheidsanalyse")

    # Load complete hexagon layer
    complete_hexagons = load_complete_hexagons('./app_data/h3_pzh_polygons.geojson')
    if complete_hexagons is None:
        st.stop()

    # Load data
    dataframes = list_and_load_csvs(CSV_FOLDER_PATH, complete_hexagons.index.tolist())

    # Sidebar selection
    with st.sidebar:
        st.header("Select Criteria")
        selected_variables = st.multiselect("Selecteer een genormalizeerde criteria voor visualisatie", list(dataframes.keys()))
        fuzz_type = st.selectbox("Selecteer of je er dichtbij wil zitten of juist ver vanaf", ["Dichtbij", "Veraf"])
        fuzz_type = 'close' if fuzz_type == "Close" else 'far'

    if selected_variables:
        # Fuzzify and collect the dataframes
        fuzzified_dataframes = []
        for var in selected_variables:
            df = dataframes[var]
            if 'value' not in df.columns:
                st.error(f"'value' column not found in dataset {var}")
                continue
            fuzzified_df = fuzzify_layer(df, fuzz_type=fuzz_type, colormap_name=COLORMAP)
            fuzzified_dataframes.append(fuzzified_df)

        # Generate Pydeck visualization
        st.pydeck_chart(generate_pydeck(fuzzified_dataframes), use_container_width=True)

        # Export buttons for each dataset
        st.header("Exporteer genormalizeerde data")
        for i, df in enumerate(fuzzified_dataframes):
            csv = df[['hex9', 'fuzzy']].to_csv(index=False).encode('utf-8')
            dataset_name = selected_variables[i]
            st.download_button(
                label=f"Download {dataset_name} as CSV",
                data=csv,
                file_name=f"{dataset_name}_fuzzified.csv",
                mime='text/csv'
            )
    else:
        st.info("Selecteer een criteria om te normalizeren.")

if __name__ == "__main__":
    main()
