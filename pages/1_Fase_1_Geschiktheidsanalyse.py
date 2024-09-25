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

# Helper function to list all CSV files in the folder
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
                dataframes[df_name] = df
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
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
    df['color'] = df[value_column].apply(
        lambda x: [int(c * 255) for c in colormap(norm(x))[:3]]
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
        apply_color_mapping(fuzzified_df, 'fuzzy', colormap)
        
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
    """Replaces underscores with spaces for cleaner display."""
    return name.replace('_', ' ')

# Categorize datasets for better user experience
def categorize_datasets(dataframes):
    """Categorizes datasets into predefined categories."""
    categories = {
        'Demografie': [
            'Aantal_inwoners',
            'Aantal_inwoners_0_tot_15_jaar',
            'Aantal_inwoners_15_tot_25_jaar',
            'Aantal_inwoners_25_tot_45_jaar',
            'Aantal_inwoners_45_tot_65_jaar',
            'Aantal_inwoners_65_jaar_en_ouder',
            'Aantal_mannen',
            'Aantal_vrouwen',
            'Gemiddelde_huishoudensgrootte',
            'Aantal_particuliere_huishoudens',
            'Aantal_eenouderhuishoudens',
            'Aantal_eenpersoonshuishoudens',
            'Aantal_meerpersoonshuishoudens_zonder_kind',
            'Aantal_tweeouderhuishoudens',
            'Aantal_personen_met_uitkering_onder_aow_leeftijd',
            'Percentage_geboren_buiten_nederland_herkomst_buiten_europa',
            'Percentage_geboren_buiten_nederland_herkomst_europa',
            'Percentage_geboren_nederland_herkomst_buiten_europa',
            'Percentage_geboren_nederland_herkomst_nederland',
            'Percentage_geboren_nederland_herkomst_overig_europa'
        ],
        'Woningvoorraad': [
            'Aantal_woningen',
            'Aantal_huurwoningen_in_bezit_van_woningcorporaties',
            'Aantal_niet_bewoonde_woningen',
            'Percentage_huurwoningen',
            'Percentage_koopwoningen',
            'Gemiddelde_woz_waarde_woning',
            'Aantal_woningen_bouwjaar_voor_1945',
            'Aantal_woningen_bouwjaar_1945_tot_1965',
            'Aantal_woningen_bouwjaar_1965_tot_1975',
            'Aantal_woningen_bouwjaar_1975_tot_1985',
            'Aantal_woningen_bouwjaar_1985_tot_1995',
            'Aantal_woningen_bouwjaar_1995_tot_2005',
            'Aantal_woningen_bouwjaar_2005_tot_2015',
            'Aantal_woningen_bouwjaar_2015_en_later',
            'Aantal_meergezins_woningen'
        ],
        'Landgebruik en Natuur': [
            'Duinen',
            'Heide',
            'Intergetijdenvlakte',
            'Landbouw',
            'Meren',
            'Moeras',
            'Natuur',
            'Natuurlijk_grasland',
            'Rivieren',
            'Sportplaats',
            'Stad',
            'Stadspark',
            'Water',
            'Weide',
            'Zee'
        ],
        'Economie en Industrie': [
            'Industrie'
        ]
    }
    # Filter out datasets that are not present
    for category in categories:
        categories[category] = [ds for ds in categories[category] if ds in dataframes.keys()]
    return categories

# Perform suitability analysis and update layers for stacked fuzzified layers
def perform_suitability_analysis_on_stack(dataframes, idx):
    categories = categorize_datasets(dataframes)

    # Create a mapping for clean names
    clean_names_map = {clean_dataset_name(name): name for name in dataframes.keys()}

    with st.sidebar.form("suitability_analysis_form"):
        st.header("Selecteer Criteria")

        selected_variables_close = []
        selected_variables_far = []

        for category, datasets in categories.items():
            with st.expander(category):
                clean_dataset_names = [clean_dataset_name(ds) for ds in datasets]
                selected_close = st.multiselect(f":one: Dichtbij ({category})", clean_dataset_names, key=f'close_{category}')
                selected_far = st.multiselect(f":two: Ver weg van ({category})", clean_dataset_names, key=f'far_{category}')
                # Map clean names back to original names and add to the lists
                selected_variables_close.extend([clean_names_map[name] for name in selected_close])
                selected_variables_far.extend([clean_names_map[name] for name in selected_far])

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
                st.warning("No significant locations found. Please adjust your criteria.")
        except Exception as e:
            st.error(f"An error occurred during spatial analysis: {e}")

    if st.sidebar.button(':two: Resultaat Opslaan & Ga naar Fase 2', help="Klik om de huidige gefilterde locaties op te slaan voor verder onderzoek in ***Fase 2: Beleid Verkenner***."):
        if 'all_loi' in st.session_state and not st.session_state.all_loi.empty:
            st.session_state.loi = st.session_state.all_loi
            st.success("Resultaten opgeslagen. U wordt doorgestuurd naar Fase 2 voor de analyse.")
            st.switch_page("pages/2_Fase_2_Beleidsverkenner.py")
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
    st.markdown("### Fase 1: Geschiktheidsanalyse")
    st.markdown("""

Welkom bij de **Geschiktheidsanalyse Tool**! Met deze tool kun je locaties vinden die het beste passen bij jouw wensen, op basis van verschillende criteria zoals bevolking, woningen, natuur en economie. De data komt uit databronnen van het CBS (Kaart van 100 meter bij 100 meter met statistieken) en de ESA (Corine Landcover Data).

### Hoe werkt het?

1. **Kies je criteria**:  
   In de **zijbalk** kies je uit categorieën zoals *Demografie*, *Woningvoorraad*, *Landgebruik en Natuur* en *Economie en Industrie*. Selecteer of je **dichtbij** of juist **ver weg** van bepaalde kenmerken wilt zijn.

2. **Bekijk data op de kaart**:  
   Klik op **"Visualiseer Dataset"** om een dataset op de kaart te zien.

3. **Voer een analyse uit**:  
   Klik op **"Bouw Geschiktheidskaart"** om een kaart te maken met gebieden die aan je gekozen criteria voldoen.

4. **Bekijk en sla resultaten op**:  
   De resultaten worden op de kaart getoond. Ben je tevreden? Sla je selectie op en ga verder naar **Fase 2** voor meer analyses.

### Waarom deze tool gebruiken?

- **Makkelijk in gebruik**: Selecteer eenvoudig de criteria die voor jou belangrijk zijn.
- **Visuele inzichten**: Bekijk de resultaten direct op een kaart.
- **Datagedreven keuzes**: Neem beslissingen gebaseerd op duidelijke data.

Veel succes met je analyse!
""")

    
    st.markdown(
        "Gebruik de dropdowns om verschillende criteria te selecteren en te analyseren.",
        unsafe_allow_html=True
    )

# Display dropdown for CSV selection and visualize the selected CSV
def display_csv_selection(dataframes):
    # Create a dictionary to map clean names to original names
    clean_names_map = {clean_dataset_name(name): name for name in dataframes.keys()}
    
    # Create categories
    categories = categorize_datasets(dataframes)
    
    with st.expander("Visualiseer Dataset"):
        for category, datasets in categories.items():
            clean_dataset_names = [clean_dataset_name(ds) for ds in datasets]
            selected_variable_clean = st.selectbox(f"Selecteer dataset ({category})", [''] + clean_dataset_names)
            if selected_variable_clean:
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
    color_map = plt.get_cmap('magma')
    apply_color_mapping(hex_df, 'value', color_map)
    return hex_df

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="Geschiktheids Analyse", layout="wide")

    # Load data
    idx = load_gdf('./app_data/h3_pzh_polygons.geojson')
    if idx is None:
        st.stop()
    all_hexagons = idx.index.tolist()
    dataframes = list_and_load_csvs(CSV_FOLDER_PATH, all_hexagons)

    # Initialize session state
    initialize_session_state(idx)

    # Display UI
    display_intro_text()
    display_csv_selection(dataframes)
    perform_suitability_analysis_on_stack(dataframes, idx)

if __name__ == "__main__":
    main()
