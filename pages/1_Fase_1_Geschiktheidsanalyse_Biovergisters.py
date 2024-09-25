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

# Constants
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'magma'
VIEW_STATE = pdk.ViewState(longitude=4.390, latitude=51.891, zoom=8)

# Helper function to list and load specified CSV files
@st.cache_data
def load_selected_csvs(folder_path, selected_csvs, all_hexagons):
    """Loads specified CSV files from the folder."""
    dataframes = {}
    for file in selected_csvs:
        file_name = f"{file}.csv"
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                if 'hex9' not in df.columns:
                    st.error(f"'hex9' column not found in {file_name}")
                    continue
                df = df.set_index('hex9')
                # Reindex to include all hexagons, fill missing values with zero
                df = df.reindex(all_hexagons, fill_value=0)
                dataframes[file] = df
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
    return dataframes

# Load GeoDataFrame from shapefile
@st.cache_data
def load_gdf(gdf_path):
    """Loads a GeoDataFrame from a shapefile."""
    gdf = gpd.read_file(gdf_path)
    if 'hex9' not in gdf.columns:
        st.error("The shapefile must contain a 'hex9' column.")
        return None
    return gdf.set_index('hex9')

def apply_color_mapping(df, value_column, colormap):
    """Applies a color map to a specified column of a DataFrame."""
    norm = plt.Normalize(vmin=df[value_column].min(), vmax=df[value_column].max())
    colormap_func = plt.get_cmap(colormap)
    df['color'] = df[value_column].apply(
        lambda x: [int(c * 255) for c in colormap_func(norm(x))[:3]]
    )  # Get RGB values

# Fuzzify input variables with "close" and "far", returning each fuzzified layer individually
def fuzzify_each_layer(df_list, fuzz_type='close', colormap_name='magma'):
    """Fuzzifies each selected criterion separately and returns a list of fuzzified DataFrames."""
    fuzzified_dataframes = []
    colormap = plt.get_cmap(colormap_name)
    
    for df in df_list:
        df_array = np.array(df['value'])
        # Avoid division by zero
        range_diff = df_array.max() - df_array.min()
        if range_diff == 0:
            range_diff = 1
        
        # Apply fuzzification depending on the fuzz_type
        if fuzz_type == "close":
            fuzzified_array = np.maximum(0, (df_array - df_array.min()) / range_diff)
        else:  # fuzz_type == "far"
            fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / range_diff)
        
        # Create a new DataFrame for the fuzzified result
        fuzzified_df = df.copy()
        fuzzified_df['fuzzy'] = np.round(fuzzified_array, 3)  # Add the fuzzified values
        
        # Apply the colormap
        apply_color_mapping(fuzzified_df, 'fuzzy', colormap_name)
        
        # Append fuzzified dataframe to the list
        fuzzified_dataframes.append(fuzzified_df.reset_index())
    
    return fuzzified_dataframes

# Stack the individual fuzzified layers into a single DataFrame
def stack_fuzzified_layers(fuzzified_dataframes):
    """Stacks multiple fuzzified DataFrames by joining them on 'hex9' index."""
    # Start with the first DataFrame in the list
    stacked_df = fuzzified_dataframes[0][['hex9', 'fuzzy']].copy()
    stacked_df.rename(columns={'fuzzy': 'fuzzy_1'}, inplace=True)  # Rename the fuzzy column to fuzzy_1

    # Add remaining fuzzified DataFrames
    for i, df in enumerate(fuzzified_dataframes[1:], start=2):
        df = df[['hex9', 'fuzzy']].copy()
        df.rename(columns={'fuzzy': f'fuzzy_{i}'}, inplace=True)
        stacked_df = pd.merge(stacked_df, df, on='hex9', how='outer')

    return stacked_df

# Spatial suitability analysis on the stacked DataFrame
def perform_spatial_analysis_on_stack(stacked_df, idx, w, g, seed=42):
    """Performs spatial suitability analysis on the stacked DataFrame with multiple fuzzified layers."""
    # Drop rows with NaN values and ensure alignment with the spatial index
    stacked_df = stacked_df.dropna(subset=stacked_df.filter(like='fuzzy').columns).set_index('hex9')
    stacked_df = stacked_df.loc[stacked_df.index.intersection(idx.index)]

    # Apply spatial weights and perform local Moran's I analysis on the stacked data
    try:
        w_subset_result = w_subset(w, stacked_df.index)
    except Exception as e:
        st.error(f"Error creating spatial weights subset: {e}")
        return pd.DataFrame()

    # Sum the fuzzified columns to get an overall score for each hexagon
    stacked_df['fuzzy_sum'] = stacked_df.filter(like='fuzzy').sum(axis=1)

    # Perform Moran's I spatial autocorrelation analysis on the summed fuzzy values
    try:
        lisa = esda.Moran_Local(stacked_df['fuzzy_sum'], w_subset_result, seed=seed)
    except Exception as e:
        st.error(f"Error performing Moran's I analysis: {e}")
        return pd.DataFrame()

    significant_locations = stacked_df[(lisa.q == 1) & (lisa.p_sim < 0.009)].index

    significant_df = stacked_df.loc[significant_locations]

        # Further filter by 'fuzzy_sum' (e.g., top 10% highest values)
    if not significant_df.empty:
        fuzzy_sum_threshold = significant_df['fuzzy_sum'].quantile(0.9)  # Adjust percentile as needed
        most_relevant_locations = significant_df[significant_df['fuzzy_sum'] >= fuzzy_sum_threshold]
    else:
        st.warning("No significant locations found after applying filters.")
        return pd.DataFrame()

    # Optionally, limit the number of results to top N
    # top_n = 100
    # most_relevant_locations = most_relevant_locations.sort_values(by='fuzzy_sum', ascending=False).head(top_n)

    return most_relevant_locations

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
    if selected_hexagons is not None and not selected_hexagons.empty:
        selected_hex_layer = pdk.Layer(
            "H3HexagonLayer",
            selected_hexagons,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            opacity=0.9,
            get_hexagon="hex9",
            get_fill_color=[142, 152, 100],  # Red color for selected hexagons
        )
        layers.append(selected_hex_layer)

    return pdk.Deck(
        initial_view_state=view_state,
        layers=layers,
        tooltip={"html": "<b>Hexagon:</b> {hex9}<br/>"}
    )

# Helper function to clean dataset names
def clean_dataset_name(name):
    """Replaces underscores with spaces and capitalizes for cleaner display."""
    return name.replace('_', ' ').capitalize()

# Perform suitability analysis and update layers for stacked fuzzified layers
def perform_suitability_analysis_on_stack(dataframes, idx):
    # Create a mapping for clean names
    clean_names_map = {clean_dataset_name(name): name for name in dataframes.keys()}

    with st.sidebar.form("suitability_analysis_form"):
        st.header("Selecteer Criteria voor Biovergisters")

        # Combine all dataset names into a single list
        clean_dataset_names = [clean_dataset_name(ds) for ds in dataframes.keys()]

        selected_close = st.multiselect(":one: Dichtbij", clean_dataset_names, key='close')
        selected_far = st.multiselect(":two: Ver weg van", clean_dataset_names, key='far')

        # Map clean names back to original names
        selected_variables_close = [clean_names_map[name] for name in selected_close]
        selected_variables_far = [clean_names_map[name] for name in selected_far]

        submit_button = st.form_submit_button("Bouw Geschiktheidskaart")

    if submit_button and (selected_variables_close or selected_variables_far):
        # Perform fuzzify on the selected datasets for "close" and "far" separately
        fuzzified_dataframes_close = fuzzify_each_layer([dataframes[var] for var in selected_variables_close], 'close', COLORMAP) if selected_variables_close else []
        fuzzified_dataframes_far = fuzzify_each_layer([dataframes[var] for var in selected_variables_far], 'far', COLORMAP) if selected_variables_far else []

        # Stack fuzzified layers
        stacked_df_close = stack_fuzzified_layers(fuzzified_dataframes_close) if fuzzified_dataframes_close else None
        stacked_df_far = stack_fuzzified_layers(fuzzified_dataframes_far) if fuzzified_dataframes_far else None

        # Combine the stacked DataFrames for 'close' and 'far' criteria
        if stacked_df_close is not None and stacked_df_far is not None:
            stacked_df = pd.merge(
                stacked_df_close, 
                stacked_df_far, 
                on='hex9', 
                how='outer',
                suffixes=('_close', '_far')
            )
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
            else:
                st.warning("Geen significante locaties gevonden. Pas je criteria aan.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden tijdens de ruimtelijke analyse: {e}")

    if st.sidebar.button('Resultaat Opslaan & Ga naar Fase 2', help="Klik om de huidige gefilterde locaties op te slaan voor verder onderzoek in Fase 2."):
        if 'all_loi' in st.session_state and not st.session_state.all_loi.empty:
            st.session_state.loi = st.session_state.all_loi
            st.success("Resultaten opgeslagen. U wordt doorgestuurd naar Fase 2 voor de analyse.")
            # st.switch_page("2_Fase_2_Beleidsverkenner.py")  # Uncomment if using multipage app
        else:
            st.error("Geen resultaten om op te slaan. Voer eerst de geschiktheidsanalyse uit.")

# Session state initialization
def initialize_session_state(idx):
    if 'w' not in st.session_state:
        st.session_state.w = weights.Queen.from_dataframe(idx, ids=idx.index.tolist())
    if 'g' not in st.session_state:
        try:
            st.session_state.g = nx.read_graphml('./app_data/G.graphml')
        except Exception as e:
            st.error(f"Error loading graph: {e}")
            st.stop()

# Display introductory text
def display_intro_text():
    st.markdown("### Geschiktheidsanalyse voor Biovergisters")
    st.markdown("""
Welkom bij de **Geschiktheidsanalyse voor Biovergisters**! Met deze tool kun je locaties vinden die het meest geschikt zijn voor nieuwe bouwprojecten, gebaseerd op verschillende criteria.

### Hoe werkt het?

1. **Kies je criteria**:  
   Gebruik de zijbalk om te selecteren op welke kenmerken je **dichtbij** of **ver weg** wilt zijn.

2. **Voer een analyse uit**:  
   Klik op **"Bouw Geschiktheidskaart"** om een kaart te maken met gebieden die aan je gekozen criteria voldoen.

3. **Bekijk en sla resultaten op**:  
   De resultaten worden op de kaart getoond. Ben je tevreden? Sla je selectie op en ga verder naar **Fase 2** voor meer analyses.

Veel succes met je analyse!
""")

# Display dropdown for CSV selection and visualize the selected CSV
def display_csv_selection(dataframes):
    # Create a dictionary to map clean names to original names
    clean_names_map = {clean_dataset_name(name): name for name in dataframes.keys()}

    with st.expander("Visualiseer Dataset"):
        clean_dataset_names = [clean_dataset_name(ds) for ds in dataframes.keys()]
        selected_variable_clean = st.selectbox("Selecteer dataset om te visualiseren", [''] + clean_dataset_names)
        if selected_variable_clean and selected_variable_clean != '':
            selected_variable = clean_names_map[selected_variable_clean]
            updated_layer = update_layer(selected_variable, dataframes)
            st.pydeck_chart(generate_pydeck(updated_layer), use_container_width=True)

# Update layer based on selected variables (non-fuzzified)
def update_layer(selected_variable, all_dataframes):
    """Updates the visualization layer with selected variable."""
    df = all_dataframes[selected_variable].reset_index()
    if 'value' not in df.columns:
        st.error(f"'value' column not found in dataset {selected_variable}")
        return pd.DataFrame()
    hex_df = df[['hex9', 'value']].copy()
    # Apply colormap
    apply_color_mapping(hex_df, 'value', COLORMAP)
    return hex_df

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="Geschiktheidsanalyse Biovergisters", layout="wide")

    # Define the selected CSV files
    selected_csvs = [
        'Industrie',
        'Natuur',
        'Stad',
        'Landbouw',
        'Water',
        'Wegen'
    ]

    # Load data
    idx = load_gdf('./app_data/h3_pzh_polygons.geojson')
    if idx is None:
        st.stop()
    all_hexagons = idx.index.tolist()
    dataframes = load_selected_csvs(CSV_FOLDER_PATH, selected_csvs, all_hexagons)

    # Initialize session state
    initialize_session_state(idx)

    # Display UI
    display_intro_text()
    display_csv_selection(dataframes)
    perform_suitability_analysis_on_stack(dataframes, idx)

if __name__ == "__main__":
    main()
