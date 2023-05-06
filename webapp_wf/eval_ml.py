from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

def evaluate_model(model, X_test, y_test):
    # Prédiction des étiquettes de classe des données de test
    predicted = model.predict(X_test)

    # Calcul de la précision du modèle en utilisant accuracy_score
    accuracy = accuracy_score(y_test, predicted)
    print("The accuracy of the model is : ", accuracy*100, "%")

    # Calcul du score F1 du modèle en utilisant f1_score
    f1 = f1_score(y_test, predicted, average='weighted')*100
    print("F1 weighted score for the model is :", f1, "%")

    # Calcul de la précision équilibrée du modèle en utilisant balanced_accuracy_score
    balanced_accuracy = balanced_accuracy_score(y_test, predicted)*100
    print("The balanced accuracy of the model is : ", balanced_accuracy, "%")

    return (accuracy, f1, balanced_accuracy)
