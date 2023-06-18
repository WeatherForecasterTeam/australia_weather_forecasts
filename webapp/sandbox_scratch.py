from utils.load_data import Dataload
from pathlib import Path
from utils.ressources import *
from utils.load_and_apply_model import *
import warnings

# Désactiver tous les avertissements
warnings.filterwarnings("ignore")

current_path = Path.cwd()
path_df_dataviz = current_path / "data" / "df_dataviz.csv"

path_df = current_path / "data" / "data_features.csv"

df = Dataload(path_df_dataviz).load_df()
# print(df.head())


path_df = current_path / "data" / "data_features.csv"
path_model = current_path / "models"


# model_random_forest = instantiate_model(RandomForestClassifier)

# train_model(model_random_forest, df_data_features)

# save_model(model_random_forest)

model_random_forest = load_model(path_model / "randomforestclassifier_model.joblib")

evaluate_model(model_random_forest, path_df)

apply_model(model_random_forest, path_df)






