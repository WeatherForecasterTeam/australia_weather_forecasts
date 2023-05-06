import streamlit as st
from ressources import *
from ressources import *
from eval_ml import *
import json
import pandas as pd
import plotly.express as px
from eval_ml import *
from run_xgb import *
from xgboost import XGBClassifier

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





# Titre de l'application
st.title("Rain in Australia")

# Création des onglets
tabs = ["Panorama", "Données", "Pluie du lendemain", "Prévision mm pluie", "Prévision long"]
page = st.sidebar.selectbox("Sélectionnez un onglet", tabs)

# Affichage du contenu de chaque onglet
if page == "Panorama":
    st.header("Vue d'ensemble des données")
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


elif page == "Données":
    st.header("Données météo")
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



elif page == "Pluie du lendemain":
    st.header("Prévision de pluie du lendemain")

    cls_df = DataLoad(r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\data\data_features.csv")  # iloc en attendant de corriger le notebook3
    json_file = r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\webapp_wf\static\models_params.json"

    # Charger les données d'entrainement
    train_data = DataLoad(r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\data\data_features.csv")



    # Séparer les données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_data.split_data_train_test()

    model, scores = train_model(json_file, *train_data.split_data_train_test())

    # model()
    df3 = pd.read_csv(r'C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\data\data_features_with_location_prepared.csv', index_col=0)


    def filter_df(df, city, date):
        day, month, year = date.split('/')
        return df[(df['location'] == city) & (df['year'] == int(year)) & (df['month'] == int(month)) & (
                    df['day'] == int(day))]


    select_ciy = filter_df(df3, "Albury", "1/12/2008").drop(columns=['raintomorrow', 'location'], axis=1)
    select_ciy2 = filter_df(df3, "Albury", "2/12/2008").drop(columns=['raintomorrow', 'location'], axis=1)

    # st.write(filter_df(df3, "Albury", "1/12/2008")[['raintomorrow', 'location', 'day']])
    # st.write(filter_df(df3, "Albury", "2/12/2008")[['raintomorrow', 'location', 'day']])

    p = model.predict(select_ciy)
    #st.write(p)
    import streamlit as st

    # Chargement du modèle
    st.title("Chargement du modèle XGBoost")

    # Affichage des scores
    f1_score = scores[0] * 100
    accuracy = scores[1] * 100
    st.subheader("Scores du modèle")
    st.text(f"F1-score (weighted) : {f1_score:.2f}%")
    st.text(f"Accuracy : {accuracy:.2f}%")
    # Description du modèle
    st.subheader("Description du modèle")
    st.markdown("""
    Le modèle XGBoost est un modèle de type "boosting" qui permet d'améliorer les prédictions en combinant plusieurs modèles faibles. 
    Ce modèle a été entraîné sur un jeu de données de 10 000 exemples avec 50 variables explicatives.
    """)


    def filter_date(df, date):
        day, month, year = date.split('/')
        return df[(df['year'] == int(year)) & (df['month'] == int(month)) & (df['day'] == int(day))]


    date_input = st.text_input('Entrer une date dans le format dd/mm/yyyy', '12/12/2008')
    df_date_select = filter_date(df3, date_input).drop(columns=['raintomorrow', 'location'], axis=1)
    df_date_select['raintomorow_predict'] = model.predict(df_date_select)


    df_date_select['raintoday'] = df_date_select['raintoday'].replace({0: 'sans pluie', 1: 'pluie'})
    df_date_select['raintomorow_predict'] = df_date_select['raintomorow_predict'].replace({0: 'sans pluie', 1: 'pluie'})


    colors_evap = {"sans pluie": 'red', "pluie": 'blue'}

    # Première carte
    fig = px.scatter_mapbox(df_date_select, lat='latitude', lon='longitude', color='raintoday',
                            color_discrete_map=colors_evap, title='Prévision')

    # Référence de tuile
    fig.update_layout(mapbox_style='carto-positron')
    # Limites de la carte et le zoom de départ
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, mapbox_center_lat=-25, mapbox_center_lon=135,
                      mapbox_zoom=2)
    st.header("Etat de la pluie du jour")
    st.plotly_chart(fig, use_container_width=True)


    # Deuxième carte
    fig = px.scatter_mapbox(df_date_select, lat='latitude', lon='longitude', color='raintomorow_predict',
                            color_discrete_map=colors_evap, title='Prévision')
    # Référence de tuile
    fig.update_layout(mapbox_style='carto-positron')

    # Limites de la carte et le zoom de départ
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, mapbox_center_lat=-25, mapbox_center_lon=135,
                      mapbox_zoom=2)
    # Afficher la carte dans Streamlit
    st.header("Prévision de la pluie du lendemain")
    st.plotly_chart(fig, use_container_width=True)


    # Utilisation du modèle pour prédire sur de nouvelles données
    # new_data = pd.read_csv('new_data.csv')
    # new_data_pred = model.predict(new_data)
    # print("Predictions: ", new_data_pred)






















elif page == "Prévision mm pluie":
    st.header("Prévision de la quantité de pluie")
    st.write("A construire")

else:
    st.header("Prévision météo à plusieurs jours")
    st.write("A construire")

if __name__ == '__main__':
    page = "Pluie du lendemain"
    st.sidebar.write("---")
    st.sidebar.write("---")
    st.sidebar.write("---")
    st.sidebar.write("© 2023 Datascientest.")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("Equipe : ")
    st.sidebar.write("    Alain BICAKCI")
    st.sidebar.write("    Samuel SIMON")
    st.sidebar.write("    Jonas LEVEQUE")
    st.sidebar.write("    Ben AYADI")