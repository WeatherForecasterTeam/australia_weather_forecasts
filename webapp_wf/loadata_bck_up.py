from ressources import *
import pandas as pd
import plotly.express as px
from eval_ml import *
json_file = r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\webapp_wf\static\models_params.json"


class DataLoad:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_df(self):
        self.df = pd.read_csv(self.file_path).iloc[:, 2:]
        return self.df

    def split_data(self):
        self.X = self.df.drop(['raintomorrow'], axis=1)
        self.y = self.df['raintomorrow']

        return self.X, self.y
    def split_data_train_test(self, test_size=0.2, random_state=42):
        self.X = self.load_df().drop(['raintomorrow'], axis=1)
        self.y = self.load_df()['raintomorrow']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def show_data(self):
        st.title("Données météo")
        st.write(self.df.head())


import pandas as pd
from sklearn.model_selection import train_test_split


cls_df = DataLoad(r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\data\df_dataviz.csv")


import streamlit as st
import plotly.express as px

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

df = cls_df.load_df()

df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

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

colors = ['navy', 'darkorange', 'green', 'green', 'green', 'darkblue', 'blue', 'purple', 'purple']


# Set the option to suppress the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define the content for the first tab
def first_tab():
    st.write("# Vue d'ensemble des données")
    # df = cls_df.load_df()
    cls_df.show_data()

    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='climat', color_discrete_map=colors_climat,
                            title='Climats des villes en Australie', hover_name='location', hover_data=['climat'])

    # Référence de tuile
    fig.update_layout(mapbox_style='carto-positron')

    # Limites de la carte et le zoom de départ
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, mapbox_center_lat=-25, mapbox_center_lon=135,
                      mapbox_zoom=2)

    # Afficher la carte dans Streamlit
    st.title("Climats des villes")
    st.plotly_chart(fig, use_container_width=True)


# Define the content for the second tab
def second_tab():
    # df = cls_df.load_df()
    st.write("# Données météo")
    st.write(" Visualisation des principales features.")

    def display_graphs10():
        df = pd.read_csv(r'C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\data\weatherAUS.csv')
        df.columns = df.columns.str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df_dateplot = df.iloc[-3000:, :]
        fig, ax = plt.subplots(figsize=[10, 5])
        plt.plot(df_dateplot['date'], df_dateplot['mintemp'], color='blue', linewidth=1, label='MinTemp')
        plt.plot(df_dateplot['date'], df_dateplot['maxtemp'], color='red', linewidth=1, label='MaxTemp')
        plt.fill_between(df_dateplot['date'], df_dateplot['mintemp'], df_dateplot['maxtemp'], facecolor='#EBF78F')
        plt.title('MinTemp vs MaxTemp by Date', fontsize=18, fontweight="bold")
        plt.legend(loc='lower left', frameon=False)
        st.pyplot(fig)
    display_graphs10()

    def display_graphs5():
        df = cls_df.load_df()
        fig, ax = plt.subplots(figsize=[10, 5])
        sns.histplot(x="humidity3pm", hue="raintomorrow", data=df, bins=50, multiple="stack")
        plt.title("Distribution de l'humidité en fonction de la prévision de pluie", fontsize=24, fontweight="bold")
        st.pyplot(fig)
    display_graphs5()

    def display_graphs6(df):
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
        fig = go.Figure(layout=go.Layout(width=900, height=800))

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
            title="Prévision météorologique du lendemain en fonction de la direction du vent la veille à 15h00"
        )

        # Display the chart using Streamlit
        st.plotly_chart(fig)

    display_graphs6(df)

    def display_graphs1():
        st.subheader("Relation MinTemp et MaxTemp avec la variable cible")
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=[20, 5])

        for i, (col, title) in enumerate(columns):
            # Clear the axis between each iteration of the loop
            ax.clear()

            sns.lineplot(data=df, x='year', y=col, color=colors[i], ax=ax)
            ax.set_title(title, fontsize=24, fontweight="bold")

            # Display the figure
            st.pyplot(fig)

    # Create the Streamlit app
    st.subheader("Evolution des variables météo par année")

    # Add a button to display or hide the graphs
    if 'show_graphs' not in st.session_state:
        st.session_state['show_graphs'] = False

    if st.session_state.show_graphs:
        display_graphs1()
        if st.button("Hide Graphs", key='1'):
            st.session_state.show_graphs = False
    else:
        if st.button("Display Graphs"):
            st.session_state.show_graphs = True




def third_tab():
    st.write("# Prévisions météo")
    st.write("Prévisions de la pluie le lendemain")
    from loadata import DataLoad
    from run_xgb import run_xgboost

    cls_df = DataLoad("../data/data_features.csv")  # iloc en attendant de corriger le notebook3

    # Charger les données
    X_train, X_test, y_train, y_test = cls_df.split_data_train_test()

    # Entraîner le modèle XGBoost
    scores = run_xgboost(json_file, X_train, X_test, y_train, y_test)

    # Afficher les scores de performance du modèle
    print(scores)




# Define the content for each tab
tabs = {
    "Panorama": first_tab,
    "Données": second_tab,
    "modele": third_tab
}

# Create the tabs
selected_tab = st.sidebar.selectbox("Sélectionner un onglet", list(tabs.keys()))
tabs[selected_tab]()


