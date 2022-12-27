def load_csv_from_github(url):
    import requests
    import base64  # pour encoder des données en base64
    import io  # pour créer un objet de type file-like à partir de données en mémoire
    import pandas as pd  # pour charger les données CSV dans un dataframe pandas

    # Créez une chaîne d'authentification en concaténant votre nom d'utilisateur et votre mot de passe
    auth_string = 'weather.forecasts.oct22@gmail.com:oct22_cds_meteo'

    # Encodez la chaîne d'authentification en utf-8
    auth_bytes = auth_string.encode('utf-8')

    # Encodez les données en base64
    auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')

    # Créez un dictionnaire de headers avec l'en-tête d'authentification
    headers = {
        'Authorization': 'Basic ' + auth_b64
    }

    # Envoyez une demande GET pour récupérer le fichier CSV
    response = requests.get(url, headers=headers)

    # Vérifiez que la réponse est correcte
    if response.status_code == 200:
        # Décodez les données de la réponse en utf-8
        csv_data = response.content.decode('utf-8')

        # Créez un objet de type file-like à partir des données de la réponse
        csv_file = io.StringIO(csv_data)

        # Chargez les données CSV dans un dataframe pandas
        df = pd.read_csv(csv_file)
        return df
    else:
        print('Error:', response.status_code)

