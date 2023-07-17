import streamlit as st
from pathlib import Path
import pandas as pd
from utils.load_data import Dataload, filter_by_date_for_model, add_city_name, process_weather_data
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
from datetime import datetime, timedelta

from pathlib import Path
# from utils.ressources import *
# from utils.load_and_apply_model import *
from utils.load_data import filter_by_date, afficher_calendrier_selection
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
df_data_features_path = current_path / "data" / "data_features_webapp.csv"
path_model = current_path / "models"
table_city = current_path / "data" / "table_city.csv"
df_chaud_humide_path = current_path / "data" /"df_chaud_humide.csv"
df_tempere_froid_path = current_path / "data" /"df_tempere_froid.csv"
df_mediterraneen_path = current_path / "data" /"df_mediterraneen.csv"
df_sec_path = current_path / "data" /"df_sec.csv"
df_local_path = current_path / "data" /"df_local.csv"

current_path = Path.cwd()
sunny_icon_path = current_path / "assets" / "images" / "sunny_2.png"
rainfall_icon_path = current_path / "assets" / "images" / "raincloud2.png"

df_climate_path = [df_chaud_humide_path, df_tempere_froid_path, df_mediterraneen_path, df_sec_path, df_local_path]
climate_model = ["randomforestclassifier_model.joblib",
"randomforestclassifier_chaud_humide_model.joblib",
"randomforestclassifier_tempere_froid_model.joblib",
"randomforestclassifier_mediterraneen_model.joblib",
"randomforestclassifier_sec_model.joblib",
"randomforestclassifier_local_model.joblib"]


# Title of the application
st.title("Rain in Australia")

tabs = ["meteo_australie",
        "analyse_exploratoire",
        "previsions_demonstration",
        "previsions_optimisation"]

# Liste des suggestions
suggestions_villes = ['Albury', 'Badgerys Creek', 'Cobar', 'Coffs Harbour', 'Moree', 'Norah Head', 'Norfolk Island',
                      'Sydney', 'Sydney Airport', 'Wagga Wagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong',
                      'Ballarat', 'Bendigo', 'Sale', 'Melbourne Airport', 'Melbourne', 'Mildura', 'Portland',
                      'Watsonia',
                      'Dartmoor', 'Brisbane', 'Cairns', 'Gold Coast', 'Townsville', 'Mount Gambier', 'Nuriootpa',
                      'Woomera',
                      'Witchcliffe', 'Pearce RAAF', 'Perth Airport', 'Perth', 'Walpole', 'Hobart', 'Launceston',
                      'Alice Springs', 'Darwin', 'Katherine', 'Uluru']


#page = st.sidebar.selectbox("Sélectionnez un onglet", tabs)
page = st.sidebar.radio("Navigation", tabs)

#Add a header and expander in side bar
st.sidebar.markdown('<p class="font">Datascientest App DS-OCT22 </p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
       Cette application a été développée dans le cadre du projet Weather Forecast, qui fait partie intégrante de notre programme de formation continue..
     """)


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

    st.title("Prévision de la pluie")
    st.header("Météo du jour")


    # Afficher le calendrier de sélection et récupérer la date sélectionnée
    date_selection = afficher_calendrier_selection()
    df_data_features = Dataload(df_data_features_path).load_df()
    st.write('Observation météo du', date_selection)
    display_map_rain(df_data_features, str(date_selection), today=True)

    if st.button("Prévision de la pluie du lendemain"):
        st.header("Prévision de la pluie du lendemain")
        model_random_forest = load_model(path_model / "randomforestclassifier_model.joblib")
        st.write('Chargement et exécution du modèle :', str(model_random_forest).split("(")[0])
        df_filtered_date = filter_by_date_for_model(df_data_features, str(date_selection))
        prediction = apply_model(model_random_forest, df_filtered_date)
        #st.write('Performance du modèle :')
        #scores = evaluate_model(model_random_forest, df_data_features_path)
        #display_scores(scores)
        date_selection_dt = datetime.strptime(date_selection, "%d/%m/%Y").date()
        st.write('\n')
        st.write('Prévision de la pluie du lendemain (', (date_selection_dt + timedelta(days=1)).strftime("%d/%m/%Y"), ')' )
        display_map_rain(prediction, str(date_selection), today=False)

        table_city = Dataload(table_city).load_df().reset_index(drop=False)

        prediction = pd.merge(prediction, table_city, on=['latitude', 'longitude'], how='left')

        filtered_cities = prediction[(prediction['raintomorrow'] == 1)]['location'].unique()

        st.write("Villes où il va pleuvoir demain :")
        for city in filtered_cities:
            st.write("    -", city)

elif page == "previsions_optimisation":

    st.title("Prévision de la pluie")
    st.header("Météo du jour")

    # Afficher le calendrier de sélection et récupérer la date sélectionnée
    date_selection = afficher_calendrier_selection()
    df_data_features = Dataload(df_data_features_path).load_df()
    st.write('Observation météo du', date_selection)
    display_map_rain(df_data_features, str(date_selection), today=True)

    st.write('Choisir une ville ou un climat :')
    # Champ de saisie avec autocomplétion
    selected_city = st.selectbox('Saisir une ville', suggestions_villes)
    # Affichage de la ville sélectionnée
    display_map_rain_with_filter(df_data_features, date_selection, selected_city, today=True)

    st.title(selected_city)
    st.write('Observation météo de la ville :', selected_city, 'le', date_selection)


    # st.write(add_city_name(df_data_features)[(add_city_name(df_data_features)['location'] == selected_city) & (
    #            add_city_name(df_data_features)['date'] == date_selection)])


    observations_selected_city = add_city_name(df_data_features)[(add_city_name(df_data_features)['location'] == selected_city) & (
                add_city_name(df_data_features)['date'] == date_selection)]

    # st.write(Dataload(df_dataviz).load_df().columns)
    df_dataviz = Dataload(df_dataviz).load_df()
    observations_selected_city_dataviz = df_dataviz[(df_dataviz['location'] == selected_city) ]
    observations_selected_city_dataviz_filtered_date = filter_by_date(observations_selected_city_dataviz, date_selection)

    date_selection = datetime.strptime(date_selection, "%d/%m/%Y")
    tomorrow_date_selection = (date_selection + timedelta(days=1))
    observations_selected_city_dataviz_filtered_date_tomorrow = filter_by_date(observations_selected_city_dataviz, str(tomorrow_date_selection))
    #st.write(observations_selected_city_dataviz_filtered_date_tomorrow)

    # observations_selected_city_dataviz['date'] = pd.to_datetime(observations_selected_city_dataviz['date'])
    # st.write(observations_selected_city_dataviz)
    # observations_selected_city_dataviz_filtered_date_yesterday = observations_selected_city_dataviz[(observations_selected_city_dataviz['date'] == previous_date_selection)]
    # st.write(observations_selected_city_dataviz_filtered_date_yesterday)
    icon_paths = [r"https://illustoon.com/photo/dl/744.png", str(rainfall_icon_path)]
    icon_paths_rain = [r"https://illustoon.com/photo/dl/2737.png", str(rainfall_icon_path)]

    col1, col2, col3, col4 = st.columns(4)
    col1.caption("Situation météo actuelle")
    if (observations_selected_city['raintoday'] == 0).all():
        col1.image(icon_paths[0], width=90)
    else:
        col1.image(icon_paths_rain[0], width=90)
    temp_city = observations_selected_city_dataviz_filtered_date['temp3pm']
    humidity_city = observations_selected_city_dataviz_filtered_date['humidity3pm']
    wind_city = observations_selected_city_dataviz_filtered_date['windspeed3pm']

    temp_city, wind_city, humidity_city = process_weather_data(observations_selected_city_dataviz_filtered_date)

    col2.metric("Température pm", str(int(temp_city)) + " °C")
    col3.metric("Vent", str(int(wind_city)) + " km/h")
    col4.metric("Humidité", str(int(humidity_city)) + " %")

    temp_city_today, wind_city_today, humidity_city_today = process_weather_data(
        observations_selected_city_dataviz_filtered_date_tomorrow)

    st.write('Observation météo de la ville :', selected_city, 'le', tomorrow_date_selection.strftime("%d/%m/%Y"))
    col1, col2, col3, col4 = st.columns(4)
    col1.caption("Situation du lendemain")
    if (observations_selected_city['raintomorrow'] == 0).all():
        col1.image(icon_paths[0], width=90)
    else:
        col1.image(icon_paths_rain[0], width=90)
    col2.metric("Température pm", str(int(temp_city_today)) + " °C")
    col3.metric("Vent", str(int(wind_city_today)) + " km/h")
    col4.metric("Humidité", str(int(humidity_city_today)) + " %")

    if st.button("Prévision de la pluie du lendemain"):
        date_selection = date_selection.strftime("%d/%m/%Y")
        st.write('\n')
        st.header("Prévision de la pluie du lendemain")
        model_random_forest = load_model(path_model / get_model(selected_city))
        model_name = get_model_name(selected_city)
        st.write('Chargement et exécution du modèle :', str(model_random_forest).split("(")[0], 'du climat', model_name)
        df_filtered_date = filter_by_date_for_model(df_data_features, str(date_selection))
        prediction = apply_model(model_random_forest, df_filtered_date)
        #st.write('Performance du modèle :')
        #scores = evaluate_model(model_random_forest, df_data_features_path)
        #display_scores(scores)
        date_selection_dt = datetime.strptime(str(date_selection), "%d/%m/%Y").date()
        st.write('\n')
        st.write('Prévision de la pluie du lendemain :', (date_selection_dt + timedelta(days=1)).strftime("%d/%m/%Y"))
        table_city = Dataload(table_city).load_df().reset_index(drop=False)

        prediction = pd.merge(prediction, table_city, on=['latitude', 'longitude'], how='left')
        prediction = prediction[prediction['location'] == selected_city]
        col1, col2  = st.columns(2)
        col1.caption("Prévision du lendemain")
        if (prediction['raintomorrow'] == 0).all():
            col1.image(icon_paths[0], width=90)
        else:
            col1.image(icon_paths_rain[0], width=90)


        temp_city_prev, wind_city_prev, humidity_city_prev = process_weather_data(observations_selected_city_dataviz_filtered_date_tomorrow)

        if (prediction['raintomorrow'] == 0).all():
            col2.write("Pas de pluie prévue")
        else:
            col1.image(icon_paths_rain[0], width=90)
            col2.write("Pluie prévue")





        filtered_cities = prediction[(prediction['raintomorrow'] == 1)]['location'].unique()


st.sidebar.title("\n")
st.sidebar.title("\n")
st.sidebar.title("\n")
st.sidebar.title("\n")
st.sidebar.title("\n")
st.sidebar.title("\n")
st.sidebar.title("\n")
st.sidebar.title("\n")
st.sidebar.title("\n")
st.sidebar.title("Equipe projet : ")
st.sidebar.write("- Jonas Levêque")
st.sidebar.write("- Alain Bicakci")
st.sidebar.write("- Samuel Simon")
st.sidebar.write("- Ben Ayadi")
