from sklearn.ensemble import RandomForestClassifier
from utils.ressources import *
from utils.load_data import Dataload
from pathlib import Path

from joblib import dump, load

# Create the complete path to the "data" folder
current_path = Path.cwd()
path_df_dataviz = current_path / "data" / "df_dataviz.csv"

path_df = current_path / "data" / "data_features.csv"

models = []



#df = Dataload(df_dataviz).load_df()
#df = df.drop(['raintomorrow'], axis=1)

# print("chemin du csv :", df_dataviz)

df = Dataload(path_df_dataviz).load_df()
# print(df.head())


path_df = current_path / "data" / "data_features.csv"
path_model = current_path / "models"

from utils.load_and_apply_model import *



## Load the model
model_random_forest = instantiate_model(RandomForestClassifier)

# Afficher les paramètres du modèle
# model_params = model_random_forest.get_params()
# print("Paramètres du modèle :")
# for param, value in model_params.items():
#     print(f"{param}: {value}")


# Load the data
# X_train, X_test, y_train, y_test = Dataload(df_data_features).split_data_train_test()

# Train the model
# model_random_forest.fit(X_train, y_train)

# Evaluate the model
# score = model_random_forest.score(X_test, y_test)
# print(f"Score du modèle : {score}")
#
#
# # Save the model
#
# # Save the model with the model name as the filename
# model_name = model_random_forest.__class__.__name__
# model_filename = f"{model_name.lower()}_model.joblib"
# dump(model_random_forest, path_model / model_filename)
#
# # Load the model
# model_random_forest = load(path_model / "randomforestclassifier_model.joblib")
#
# # Evaluate the model
# score = model_random_forest.score(X_test, y_test)
#
# print(f"Score du modèle : {score}")
#



train_model(model_random_forest, path_df)

save_model(model_random_forest)

model_random_forest = load_model(path_model / "randomforestclassifier_model.joblib")

# evaluate_model(model_random_forest, df_data_features)

