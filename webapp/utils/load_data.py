from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

current_path = Path.cwd()
path_df_dataviz = current_path.parent / "data" / "df_dataviz.csv"

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

def filter_by_date(df, date_du_jour):
    # Convertir les colonnes "annee", "mois" et "jour" en type entier
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    date_du_jour = pd.to_datetime(date_du_jour, format='%d/%m/%Y')
    df_filtered = df[df['date'] == date_du_jour]
    return df_filtered




if __name__ == '__main__':
    print(path_df_dataviz)
    Dataload(path_df_dataviz).load_df()
