from ressources import *
import streamlit as st

# Titre de l'application
st.title("Weather forecast project")

# Sous-titre
st.header("Projet Datascientest")

# Zone de saisie de texte
nom_utilisateur = st.text_input("Entrez votre nom")

# Bouton de soumission
if st.button("Soumettre"):
    st.success("Bonjour, {}!".format(nom_utilisateur))


# Générer des données aléatoires
np.random.seed(1)
x = np.arange(0, 10, 0.1)
y = np.random.randn(len(x))

# Créer un graphique Plotly
fig = go.Figure(data=go.Scatter(x=x, y=y))

# Ajouter des titres et des étiquettes d'axes
fig.update_layout(
    title="Exemple de graphique Plotly",
    xaxis_title="X",
    yaxis_title="Y"
)

# Afficher le graphique en utilisant la fonction st.plotly_chart
st.plotly_chart(fig, use_container_width=True)

# charger les données
df = pd.read_csv("../data/data_features.csv").iloc[:,2:] #iloc en attendant de corriger le notebook3

# Séparer les données en variables explicatives et variable cible
X = df.drop(['raintomorrow'],axis=1)
y = df['raintomorrow']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10)

def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0 = time.time()
    if verbose == False:
        model.fit(X_train, y_train, verbose=0)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time() - t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test, y_pred, digits=5))

    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    fper, tper, thresholds = roc_curve(y_test, probs)
    plot_roc_cur(fper, tper)

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.show()

    return model, accuracy, roc_auc, coh_kap, time_taken


def plot_roc_cur(fper, tper):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)

params = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # step for each iteration
    'silent': 1, # keep it quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3, # the number of classes
    'eval_metric': 'merror'} # evaluation metric

num_round = 20  # the number of training iterations (number of trees)


model = xgb.train(params,
                  train,
                  num_round,
                  verbose_eval=2,
                  evals=[(train, 'train')])


preds = model.predict(test)
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Precision: {:.2f} %".format(precision_score(y_test, best_preds, average='macro')*100))


xgbc = XGBClassifier(objective='binary:logistic')
xgbc.fit(X_train,y_train)
predicted = xgbc.predict(X_test)
print ("The accuracy of Logistic Regression is : ", accuracy_score(y_test, predicted)*100, "%")
print()
print("F1 score for XGBoost is :",f1_score(y_test, predicted,)*100, "%")


params_xgb ={'n_estimators': 500,
            'max_depth': 16}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, coh_kap_xgb, tt_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)



