from ressources import *
from eval_ml import *
import json
from xgboost import XGBClassifier


json_file = r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\webapp_wf\static\models_params.json"


def run_xgboost(json_file, X_train, X_test, y_train, y_test):
    # Charger les paramètres depuis le fichier JSON
    with open(json_file, 'r') as f:
        params = json.load(f)['XGBClassifier']

    # Créer un modèle XGBoost avec les paramètres spécifiés
    model = XGBClassifier(**params)

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Évaluer les performances du modèle sur les données de test
    scores = evaluate_model(model, X_test, y_test)

    # Retourner les scores de performance du modèle
    return scores

import json
import joblib
from xgboost import XGBClassifier

def train_model(json_file, X_train, X_test, y_train, y_test):
    # Charger les paramètres depuis le fichier JSON
    with open(json_file, 'r') as f:
        params = json.load(f)['XGBClassifier']

    # Créer un modèle XGBoost avec les paramètres spécifiés
    model = XGBClassifier(**params)

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Évaluer les performances du modèle sur les données de test
    scores = evaluate_model(model, X_test, y_test)

    # Retourner le modèle entraîné et les scores de performance sur les données de test
    return model, scores

def save_model(model, filename):
    # Sauvegarder le modèle entraîné en utilisant joblib
    joblib.dump(model, filename)

def load_model(filename):
    # Charger le modèle entraîné depuis le fichier
    model = joblib.load(filename)
    return model

def apply_model(model, X):
    # Utiliser le modèle pour prédire les sorties pour de nouvelles données
    predictions = model.predict(X)
    prediction_scores = model.predict_proba(X)[:, 1]
    return predictions, prediction_scores

from sklearn.metrics import classification_report

def evaluate_model(model, X, y):
    # Évaluer les performances du modèle sur de nouvelles données X
    y_pred = model.predict(X)
    accuracy = model.score(X, y)
    report = classification_report(y, y_pred, output_dict=True)
    print('Accuracy:', accuracy)
    f1_weighted = report['weighted avg']['f1-score']
    print('F1-score (weighted):', f1_weighted)
    return accuracy, f1_weighted




if __name__ == "__main__":
    from loadata import DataLoad
    from ressources import *

    json_file = r"C:\Users\benme\Documents\datascientest\projet\australia_weather_forecasts\webapp_wf\static\models_params.json"

    cls_df = DataLoad("../data/data_features.csv")  # iloc en attendant de corriger le notebook3

    # Charger les données
    X_train, X_test, y_train, y_test = cls_df.split_data_train_test()

    # Entraîner le modèle XGBoost
    scores = run_xgboost(json_file, X_train, X_test, y_train, y_test)

    # Afficher les scores de performance du modèle
    print(scores)