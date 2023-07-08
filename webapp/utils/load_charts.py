import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
from utils.load_data import filter_by_date, add_city_name
from datetime import datetime, timedelta


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

current_path = Path.cwd()
sunny_icon_path = current_path / "assets" / "images" / "sunny-svg-2.png"
rainfall_icon_path = current_path / "assets" / "images" / "raincloud2.png"


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


def display_map_rain_model(df, day):
    # Création de la carte centrée sur l'Australie
    map = folium.Map(location=[-25, 135], zoom_start=3)

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


def display_map_rain(df_in, day, today=True):
    # Création de la carte centrée sur l'Australie avec le fond de carte Stamen Terrain
    df_filter = filter_by_date(df_in, day)
    df_filter = add_city_name(df_filter)
    # Carte centrée sur l'Australie
    m = folium.Map(location=[-25.2744, 133.7751], zoom_start=3)

    if today:
        # Déterminer les icônes en fonction de la valeur de raintomorrow
        conditions = [
            df_filter['raintoday'] == 0,
            df_filter['raintoday'] == 1
        ]
    else:
        # Déterminer les icônes en fonction de la valeur de raintomorrow
        conditions = [
            df_filter['raintomorrow'] == 0,
            df_filter['raintomorrow'] == 1
        ]
    icon_paths = [str(sunny_icon_path), str(rainfall_icon_path)]

    # Ajouter les marqueurs à la carte avec les icônes personnalisées
    for condition, icon_path in zip(conditions, icon_paths):
        filtered_data = df_filter[condition]
        for _, row in filtered_data.iterrows():
            lat = row['latitude']  # Latitude
            lon = row['longitude']  # Longitude
            location = row['location']  # Nom de la ville
            icon = folium.CustomIcon(icon_image=icon_path, icon_size=(40, 40))
            marker = folium.Marker(location=[lat, lon], icon=icon)
            marker.add_child(folium.Tooltip(location))
            marker.add_to(m)

    # Afficher la carte dans le notebook
    return folium_static(m)

def display_map_rain_with_filter(df_in, day, city_or_climat, today=True):
    # Création de la carte centrée sur l'Australie avec le fond de carte Stamen Terrain
    df_filter = filter_by_date(df_in, day)
    df_filter = add_city_name(df_filter)
    df_filter = df_filter[df_filter['location'] == city_or_climat]
    # Carte centrée sur l'Australie
    m = folium.Map(location=[-25.2744, 133.7751], zoom_start=3)

    if today:
        # Déterminer les icônes en fonction de la valeur de raintomorrow
        conditions = [
            df_filter['raintoday'] == 0,
            df_filter['raintoday'] == 1
        ]
    else:
        # Déterminer les icônes en fonction de la valeur de raintomorrow
        conditions = [
            df_filter['raintomorrow'] == 0,
            df_filter['raintomorrow'] == 1
        ]
    icon_paths = [str(sunny_icon_path), str(rainfall_icon_path)]

    # Ajouter les marqueurs à la carte avec les icônes personnalisées
    for condition, icon_path in zip(conditions, icon_paths):
        filtered_data = df_filter[condition]
        for _, row in filtered_data.iterrows():
            lat = row['latitude']  # Latitude
            lon = row['longitude']  # Longitude
            location = row['location']  # Nom de la ville
            icon = folium.CustomIcon(icon_image=icon_path, icon_size=(40, 40))
            marker = folium.Marker(location=[lat, lon], icon=icon)
            marker.add_child(folium.Tooltip(location))
            marker.add_to(m)

    # Afficher la carte dans le notebook
    return folium_static(m)

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
# colors = {'coastal': 'blue',
#           'valley': 'green',
#           'mountainous': 'purple',
#           'desert': 'darkgoldenrod',
#           'island': 'skyblue'}


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

def afficher_calendrier_selection():
    # Définir les dates minimale et maximale
    date_min = datetime(2008, 12, 1)
    date_max = datetime(2016, 4, 26)

    # Afficher le sélecteur de date dans Streamlit et récupérer la date sélectionnée
    selected_date = st.date_input("Sélectionnez une date", value=datetime(2010, 8, 20), min_value=date_min, max_value=date_max)

    # Convertir la date sélectionnée en format souhaité (%d/%m/%Y)
    selected_date_str = selected_date.strftime("%d/%m/%Y")

    # Retourner la date sélectionnée au format souhaité
    return selected_date_str
