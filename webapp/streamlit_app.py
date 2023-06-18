import streamlit as st
from pathlib import Path
import pandas as pd
from utils.load_data import Dataload
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_charts import *
from IPython.display import Image
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import os

from pathlib import Path
# from utils.ressources import *
# from utils.load_and_apply_model import *
from utils.load_data import filter_by_date
from utils.load_and_apply_model import *

# Set basic page config
st.set_page_config(
    page_title="Weather forecast project",
    page_icon='🌧️',
    layout='centered'
)

# Retrieve the absolute path of the current directory
current_path = Path.cwd()

# Create the complete path to the "data" folder
df_dataviz = current_path / "data" / "df_dataviz.csv"
df_city = current_path / "data" / "data_features_city.csv"
path_images = current_path / "assets" / "images" / "features"
df_data_features = current_path / "data" / "data_features.csv"
path_model = current_path / "models"

# Title of the application
st.title("Rain in Australia")

tabs = ["meteo_australie",
        "analyse_exploratoire",
        "previsions_demonstration",
        "conclusion_perspectives"]

page = st.sidebar.selectbox("Sélectionnez un onglet", tabs)

if page == "meteo_australie":
    st.write("""
    En Australie, le régime des précipitations est fortement lié aux saisons. Comparée aux autres masses continentales, l'Australie est l'une des plus arides. Plus de 80 % de sa surface connaît une pluviométrie annuelle inférieure à 600 millimètres.

    En passant à un autre extrême, certaines côtes nord du Queensland ont des moyennes annuelles de plus de 4 000 mm, le record australien de 12 461 mm ayant été atteint en 2000 au sommet du Mount Bellenden Ker, qui possède aussi depuis 1986 le record mondial de précipitations en 8 jours avec 3 847 mm.

    En Australie, le régime des précipitations est fortement lié aux saisons. Comparée aux autres masses continentales, l'Australie est l'une des plus arides. Plus de 80 % de sa surface connaît une pluviométrie annuelle inférieure à 600 millimètres.

    En passant à un autre extrême, certaines côtes nord du Queensland ont des moyennes annuelles de plus de 4 000 mm, le record australien de 12 461 mm ayant été atteint en 2000 au sommet du Mount Bellenden Ker, qui possède aussi depuis 1986 le record mondial de précipitations en 8 jours avec 3 847 mm.
    """)

    st.write("""Objectif du projet : 

    1 - Préparer les données météorologiques de l'Australie pour l'analyse 
        et la modélisation.

    2 - Entrainer des modèles de prévision de la pluie en fonction 
        des climats de l'Australie.

    3 - Prédire la pluie du lendemain en Australie en fonction 
        des données météorologiques du jour.""")

    st.image('http://www.carte-du-monde.net/pays/australie/carte-climat-australie.jpg')


elif page == 'analyse_exploratoire':
    st.header("Analyse exploratoire des données météo")

    data_loader = Dataload(df_dataviz)  # Create an instance of the Dataload class with the file path
    df = data_loader.load_df()  # Load the data into a dataframe

    st.write(df.head())  # Display the first few rows of the dataframe

    display_map_climat(df)

    plot_rain_forecast(df)

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    colors = ['navy', 'darkorange', 'green', 'green', 'green', 'darkblue', 'blue', 'purple', 'purple']

    # Set the option to suppress the PyplotGlobalUseWarning
    st.set_option('deprecation.showPyplotGlobalUse', False)

    display_radar(df)

    plot_humidity_distribution(df)

    if st.button("Afficher les features"):
        # Appelez la fonction display_features_png avec les paramètres appropriés
        display_features_png(path_images)


elif page == "previsions_demonstration":
    st.header("Modèles de prévision de la pluie")

    df = Dataload(df_city).load_df()

    display_map_rain(filter_by_date(df, "05/05/2014"), 'raintoday')

    model_random_forest = load_model(path_model / "randomforestclassifier_model.joblib")

    evaluate_model(model_random_forest, df_data_features)

    apply_model(model_random_forest, df_data_features)


