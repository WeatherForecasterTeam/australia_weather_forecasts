import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from PIL import Image
import os
import pandas as pd

import seaborn as sns
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path


# Définition des couleurs pour chaque type de géographie
colors = {'coastal': 'blue',
          'valley': 'green',
          'mountainous': 'purple',
          'desert': 'darkgoldenrod',
          'island': 'skyblue'}

colors_climat = {'Chaud humide': 'green',
                 'Tempéré froid': 'blue',
                 'Sec': '#ffea00',
                 'Méditerranéen': 'orange'}
colors_rain = {0: 'red',
               1: 'blue'}


def display_map_climat(df):
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='climat', color_discrete_map=colors_climat,
                            title='Climats des villes en Australie', hover_name='location', hover_data=['climat'])
    # Référence de tuile
    fig.update_layout(mapbox_style='carto-positron')

    # Limites de la carte et le zoom de départ
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, mapbox_center_lat=-25, mapbox_center_lon=135,
                      mapbox_zoom=2)

    # Augmentation de la taille des points
    fig.update_traces(marker=dict(size=10))

    # Afficher la carte dans Streamlit
    st.title("Climats des villes")
    st.plotly_chart(fig, use_container_width=True)


def display_map_rain(df, day):
    # Création de la carte centrée sur l'Australie
    map = folium.Map(location=[-25, 135], zoom_start=4)

    # Parcourir chaque ligne du DataFrame
    for index, row in df.iterrows():
        # Obtenir les coordonnées de la localisation
        latitude = row['latitude']
        longitude = row['longitude']

        # Obtenir la valeur de day
        rain = row[day]

        # Déterminer la couleur du marqueur en fonction de la valeur de day
        color = 'blue' if rain == 1 else 'red'

        # Créer le marqueur et l'ajouter à la carte
        folium.Marker(location=[latitude, longitude], icon=folium.Icon(color=color)).add_to(map)

    # Afficher la carte dans Streamlit
    folium_static(map)


def display_radar(df):
    # Labels pour chaque axe du graphe en mode radar
    labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW',
              'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # Sélection des données pour les jours avec pluie
    df_rain = df[df['raintomorrow'] == 'Yes']

    # Liste de données pour les jours avec pluie
    rain_data = df_rain['winddir3pm'].value_counts()

    # Sélection des données pour les jours sans pluie
    df_no_rain = df[df['raintomorrow'] == 'No']

    # Liste de données pour les jours sans pluie
    no_rain_data = df_no_rain['winddir3pm'].value_counts()

    # Création d'un objet Figure
    fig = go.Figure(layout=go.Layout(width=600, height=600))

    # Ajout des données pour les jours sans pluie au graphe en mode radar
    fig.add_trace(go.Scatterpolar(
        r=no_rain_data,
        theta=labels,
        fill='toself',
        name='Jour sans pluie',
        marker=dict(color='orange')
    ))

    # Ajout des données pour les jours avec pluie au graphe en mode radar
    fig.add_trace(go.Scatterpolar(
        r=rain_data,
        theta=labels,
        fill='toself',
        name='Jour avec pluie',
        marker=dict(color='blue')
    ))

    # Ajustement de l'échelle des axes du graphe en mode radar
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(rain_data.max(), no_rain_data.max())]
            )),
        showlegend=True,
        title="Pluie du lendemain en fonction de la direction du vent la"
              " veille à 15h00"
    )

    # Display the chart using Streamlit
    st.plotly_chart(fig)


# Définition des couleurs pour chaque type de géographie
colors = {'coastal': 'blue',
          'valley': 'green',
          'mountainous': 'purple',
          'desert': 'darkgoldenrod',
          'island': 'skyblue'}


def display_features(df, path_images):
    # Features
    columns = [('rainfall', "Rain by year"),
               ('evaporation', "Evaporation by year"),
               ('windgustspeed', 'WindGustSpeed'),
               ('windspeed9am', 'Wind Speed 9AM'),
               ('windspeed3pm', 'Wind Speed 3PM'),
               ('humidity9am', 'Humidity 9AM'),
               ('humidity3pm', 'Humidity 3PM'),
               ('temp9am', 'Temperature 9AM'),
               ('temp3pm', 'Temperature 3PM')]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=[20, 5])

    # Define the colors dictionary
    colors = {'rainfall': 'blue',
              'evaporation': 'green',
              'windgustspeed': 'purple',
              'windspeed9am': 'darkgoldenrod',
              'windspeed3pm': 'skyblue',
              'humidity9am': 'orange',
              'humidity3pm': 'red',
              'temp9am': 'pink',
              'temp3pm': 'brown'}

    for i, (col, title) in enumerate(columns):
        # Clear the axis between each iteration of the loop
        ax.clear()

        sns.lineplot(data=df, x='date', y=col, color=colors[col], ax=ax)
        ax.set_title(title, fontsize=24, fontweight="bold")

        # Save the figure
        filename = f"{col}.jpg"
        filepath = path_images / filename
        fig.savefig(str(filepath))  # Convertir le chemin en chaîne

        # Display the figure
        st.image(str(filepath))  # Convertir le chemin en chaîne



def plot_rain_forecast(df):
    df_group = df.groupby(['climat', 'raintomorrow']).size().reset_index(name='counts')

    fig = px.bar(df_group, y='climat', x='counts', color='raintomorrow',
                 title='Prévision de pluie par zone climatique',
                 labels={'counts': 'Nombre de prévisions', 'climat': 'Zone climatique'},
                 template='plotly_white')

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=14),
        height=400  # Définir la taille du graphique en pixels
    )

    st.plotly_chart(fig)


def display_features_png(path_images):
    # Récupération des chemins des fichiers PNG dans le répertoire spécifié
    images_path = Path(path_images)
    png_files = list(images_path.glob('*.jpg'))

    # Affichage des images PNG et de leur chemin
    for filepath in png_files:
        image = Image.open(filepath)
        st.image(image, use_column_width=True)

def plot_humidity_distribution(df):
    fig = px.histogram(df, x="humidity3pm", color="raintomorrow", nbins=100, barmode="overlay")
    fig.update_layout(
        title={
            "text": "Distribution de l'humidité en fonction de la prévision de pluie",
            "font": {"size": 24}
        },
        xaxis_title="Humidité à 15h",
        yaxis_title="Nombre d'échantillons"
    )
    st.plotly_chart(fig)