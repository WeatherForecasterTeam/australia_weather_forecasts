import streamlit as st
import pickle
import pandas as pd
from xgboost import XGBClassifier

from loadata import DataLoad
from run_xgb import run_xgboost


def train_and_save_model():
    # Charger les données
    cls_df = DataLoad(
        r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\data\data_features.csv")  # iloc en attendant de corriger le notebook3
    X_train, X_test, y_train, y_test = cls_df.split_data_train_test()

    # Entraîner le modèle XGBoost
    json_file = r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\webapp_wf\static\models_params.json"
    model = run_xgboost(json_file, X_train, X_test, y_train, y_test)

    # Sauvegarder le modèle
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Afficher les scores de performance du modèle
    scores = model
    #print(scores)
    st.write("Le modèle est prêt")


def load_and_predict(id_city, model_file):
    # Charger le modèle depuis le fichier pickle
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Charger les données pour la ville donnée
    cls_df = DataLoad(
        r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\data\data_features.csv")
    df = cls_df.load_df()
    X = cls_df.get_data_for_city(id_city)
    X_pred = X[:, :-1]
    y_pred = X[:, -1]

    json_file = r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\webapp_wf\static\models_params.json"

    cls_df = DataLoad(
        r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\data\data_features.csv")  # iloc en attendant de corriger le notebook3

    # Hyper paramètres optimisés
    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        max_depth=5,
        learning_rate=0.1,
        gamma=0.1,
        subsample=0.5,
        colsample_bytree=0.8,
        eval_metric='merror',
        min_child_weight=3,
        n_estimators=200
    )


    y_pred = model.predict(X_pred)



    # Afficher les prédictions
    st.write(f"La pluie du lendemain pour {id_city} est de : {y_pred[0]:.2f} mm")


# Interface utilisateur Streamlit
st.title("Prévisions de la pluie en Australie")

# Menu principal
menu = ["Accueil", "Pluie du lendemain"]
choice = st.sidebar.selectbox("Sélectionnez une option", menu)

if choice == "Accueil":
    st.subheader("Bienvenue sur notre application de prévisions de la pluie en Australie.")
    st.write("Veuillez sélectionner une option dans la barre latérale.")

elif choice == "Pluie du lendemain":
    st.header("Pluie du lendemain")

    # Entraîner et sauvegarder le modèle si nécessaire
    if st.button("Entraîner le modèle"):
        train_and_save_model()

    # Charger le modèle et faire des prédictions
    model_file = 'model.pkl'
    city_name = st.text_input("Entrez le nom de la ville")
    if st.button("Prédire la pluie du lendemain"):
        load_and_predict(city_name, model_file)

if __name__ == '__main__':
    #pass
    #train_and_save_model()
    model_file = r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\webapp_wf\model.pkl"
    load_and_predict(1, model_file)
