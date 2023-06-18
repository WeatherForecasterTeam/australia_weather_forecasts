from sklearn.metrics import classification_report
from joblib import dump, load
from pathlib import Path
import json

from utils.load_data import Dataload


PARAM_MODELS = Path.cwd().parent / "webapp" / "models" / "models_params.json"
path_model = Path.cwd().parent / "webapp" / "models"
path_df = Path.cwd().parent / "webapp" / "data" / "data_features.csv"


def instantiate_model(model_name):
    # Charger les paramètres depuis le fichier JSON
    with open(PARAM_MODELS, 'r') as f:
        params = json.load(f)[model_name.__name__]

    # Créer un modèle avec les paramètres spécifiés
    model = model_name(**params)

    # Retourner le modèle
    return model


def train_model(model, dataset):
    # Séparer les données en entrées et étiquettes de classe
    X_train, X_test, y_train, y_test = Dataload(dataset).split_data_train_test()
    # Entraîner le modèle
    model.fit(X_train, y_train)
    # Retourner le modèle entraîné
    print("Model trained")
    return model


def save_model(model):
    # Sauvegarder le modèle entraîné en utilisant joblib
    model_name = model.__class__.__name__
    model_filename = f"{model_name.lower()}_model.joblib"
    dump(model, path_model / model_filename)
    print("Model saved")


def load_model(filename):
    # Charger le modèle entraîné depuis le fichier
    model = load(filename)
    print("Model loaded")
    return model


# noinspection PyTypeChecker
def evaluate_model(model, dataset):
    X_train, X_test, y_train, y_test = Dataload(dataset).split_data_train_test()
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print('Accuracy:', f'{accuracy*100:.2f} %')
    f1_weighted = report['weighted avg']["f1-score"]
    print('F1-score (weighted):', f'{f1_weighted*100:.2f} %')
    print("evaluate_model done")
    return accuracy, f1_weighted


def apply_model(model, dataset):
    X_train, X_test, y_train, y_test = Dataload(dataset).split_data_train_test()
    predictions = model.predict(X_test)
    prediction_scores = model.predict_proba(X_test)[:, 1]
    print("apply_model done")
    return predictions, prediction_scores
