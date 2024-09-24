import streamlit as st

def main():
    st.set_page_config(page_title="Begin", layout="wide", initial_sidebar_state="expanded")

    st.title(":seedling: Multi Criteria analyse, Een Interactieve Tool")
    
    st.markdown(":bulb: Over de tool...")
    st.markdown("Voor de duidelijkheid: deze tool is bedoelt als startgesprek voor verschillende stakeholders, niet als leidend in besluitvorming. De tool bestaat uit een ***tweestaps*** leerproces om gebruikers te betrekken bij het leren over de voordelen en afwegingen van een bepaald scenario.")

    st.markdown(":compass: Hoe de tool te gebruiken...")
    st.markdown("**Fase 1: Geschiktheidsanalyse**")
    st.markdown("Fase 1 verwelkomt gebruikers om een ​​geschiktheidsanalyse op meerdere criteria uit te voeren. Geschiktheidsanalyse kan worden beschouwd als een methode voor locatieselectie.")
    st.markdown("**Fase 2: Beleidsverkenner**")
    st.markdown("Fase 2 nodigt gebruikers uit om combinaties van kandidaat-locaties voor een scenario te verkennen.")
    st.markdown("We zullen uw lijst met kandidaatlocaties uit Fase 1 gebruiken om scenario's te genereren die bestaan ​​uit de meest strategische locaties voor een specifiek scenario.")
    st.markdown("Aan het einde van deze fase leer je de kosten en baten van verschillende scenario’s.")
    st.markdown(":repeat: **Iterative Learning**") 
    col1, col2, col3 = st.columns(3)
    with col2: 
        st.image("./two_phase.png")

    st.markdown("Deze tool is ontwikkeld voor Data Gedreven werken binnen het Provincie Zuid Holland Programma Landelijk Gebied. Het bouwt verder op het al bestaande BIOZE project. BIOZE is ontwikkeld voor het EU Interreg Project: BIOmass skills for Net Zero (BIOZE), door de Faculteit Geo-Informatie Wetenschap en Aardobservatie (ITC) van de Universiteit Twente.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
