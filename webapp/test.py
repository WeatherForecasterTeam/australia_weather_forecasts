from sklearn.ensemble import RandomForestClassifier
from utils.ressources import *
from utils.load_data import Dataload
from pathlib import Path
from utils.load_and_apply_model import *
from utils.load_data import filter_by_date

from joblib import dump, load

# Create the complete path to the "data" folder
current_path = Path.cwd()
path_df_dataviz = current_path / "data" / "df_dataviz.csv"
path_model = current_path / "models"
df_data_features = current_path / "data" / "data_features_webapp.csv"


model_random_forest = instantiate_model(RandomForestClassifier)



df_data_features = Dataload(df_data_features).load_df()

train_model(model_random_forest, path_df)

save_model(model_random_forest)

model_random_forest = load_model(path_model / "randomforestclassifier_model.joblib")

evaluate_model(model_random_forest, path_df)

df_filtered_date = filter_by_date(path_df, str("02/05/2010"))

apply_model(model_random_forest, df_filtered_date)

prediction = apply_model(model_random_forest, df_filtered_date)

print(prediction.columns)