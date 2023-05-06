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

    def load_data_city(self):
        self.df = pd.read_csv(self.file_path).iloc[:, :]
        return self.df

    def split_data(self):
        self.X = self.df.drop(['raintomorrow'], axis=1)
        self.y = self.df['raintomorrow']
        return self.X, self.y

    def split_data_train_test(self, test_size=0.2, random_state=42):
        self.load_df()
        self.X = self.df.drop(['raintomorrow'], axis=1)
        self.y = self.df['raintomorrow']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data_for_city(self, id_city):
        self.load_df()
        city_data = self.df.loc[self.df.index == id_city]
        return city_data

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
