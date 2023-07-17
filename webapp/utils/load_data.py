from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta


current_path = Path.cwd()
path_df_dataviz = current_path.parent / "data" / "df_dataviz.csv"
table_city = current_path / "data" / "table_city.csv"

#test commit ²
class Dataload:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_validation = None
        self.y_validation = None

    def load_df(self):
        self.df = pd.read_csv(self.file_path, index_col=0)
        return self.df

    def load_data_city(self):
        self.df = pd.read_csv(self.file_path, sep=',')
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

    def split_data_train_test_validation(self, test_size=0.2, val_size=0.25, random_state=42):
        self.load_df()
        self.X = self.df.drop(['raintomorrow'], axis=1)
        self.y = self.df['raintomorrow']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=random_state)
        return self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation

    def get_data_for_city(self, id_city):
        self.load_df()
        city_data = self.df.loc[self.df.index == id_city]
        return city_data

    def show_data(self):
        st.title("Données météo")
        st.write(self.df.head())

def filter_by_date_for_model(df_dd, date_du_jour, return_index = False):
    # Convertir les colonnes "annee", "mois" et "jour" en type entier
    if type(df_dd) is not pd.DataFrame:
        df_dd = pd.read_csv(df_dd, index_col=0)
    df_dd['year'] = df_dd['year'].astype(int)
    df_dd['month'] = df_dd['month'].astype(int)
    df_dd['day'] = df_dd['day'].astype(int)

    df_dd['date'] = pd.to_datetime(df_dd[['year', 'month', 'day']])
    date_du_jour = pd.to_datetime(date_du_jour, format='%d/%m/%Y')
    df_filtered = df_dd[df_dd['date'] == date_du_jour]
    df_filtered = df_filtered.drop(['raintomorrow'], axis=1)
    indexes = df_filtered.index
    if return_index:
        return indexes
    else:
        return df_filtered

def filter_by_date(df, date_du_jour):
    # Convertir les colonnes "annee", "mois" et "jour" en type entier
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    date_du_jour = pd.to_datetime(date_du_jour)
    df_filtered = df[df['date'] == date_du_jour]

    return df_filtered

def add_city_name(df):
    df_city = Dataload(table_city).load_df().reset_index(drop=False)
    df = pd.merge(df, df_city, on=['latitude', 'longitude'], how='left')
    return df

import pandas as pd

def process_weather_data(observations):
    temp_city = observations['temp3pm'] if 'temp3pm' in observations else observations['temp_amplitude']
    humidity_city = observations['humidity3pm']
    wind_city = observations['windspeed3pm']

    if not temp_city.empty:
        temp_city_value = temp_city.iloc[0]
    else:
        temp_city_value = 17  # Default temperature set to 17°C if no data

    if not humidity_city.empty:
        humidity_city_value = humidity_city.iloc[0]
    else:
        humidity_city_value = 30  # Default humidity set to 30% if no data

    if not wind_city.empty:
        wind_city_value = wind_city.iloc[0]
    else:
        wind_city_value = 9  # Default wind speed set to 9 if no data

    return temp_city_value, humidity_city_value, wind_city_value


def afficher_calendrier_selection():
    # Définir les dates minimale et maximale
    date_min = datetime(2008, 12, 1)
    date_max = datetime(2016, 4, 26)

    # Afficher le sélecteur de date dans Streamlit et récupérer la date sélectionnée
    selected_date = st.date_input("Sélectionnez une date", value=datetime(2008, 12, 30), min_value=date_min, max_value=date_max)

    # Convertir la date sélectionnée en format souhaité (%d/%m/%Y)
    selected_date_str = selected_date.strftime("%d/%m/%Y")

    # Retourner la date sélectionnée au format souhaité
    return selected_date_str




if __name__ == '__main__':
    print(path_df_dataviz)
    Dataload(path_df_dataviz).load_df()
