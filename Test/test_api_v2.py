# # test_api.py
import sys
import os
import pytest
import json
import requests

URL_API_HOST = "http://localhost:5001/"  # Utilisation en local
URL_API = "http://projet7API:5001/"  # Utilisation en production
URL_DASHBOARD = "http://localhost:8501/"

# Fonction de test pour la route infos_gen
def test_site_prod_running():
    response = requests.get('https://p7.wdavid.chevaliers.oublies.fr/')
    print(response.status_code)
    assert response.status_code == 200  # Vérifiez que la réponse est HTTP 200 (OK)
# Fonction de test pour la route infos_gen
def test_site_test_running():
    response = requests.get("http://localhost:8501/")
    print(response.status_code)
    assert response.status_code == 200  # Vérifiez que la réponse est HTTP 200 (OK)

# def test_infos_gen():
#     response = requests.get(URL_API_HOST + 'infos_gen')
#     print(response.status_code)
#     assert response.status_code == 200  # Vérifiez que la réponse est HTTP 200 (OK)

# Fonction de test pour la route infos_gen
def test_infos_gen():
    response = requests.get(URL_API_HOST + 'infos_gen')
    print(response.status_code)
    assert response.status_code == 200  # Vérifiez que la réponse est HTTP 200 (OK)

    # Obtenez les données de la réponse au format JSON
    data = response.json()

    # Affichez les données dans le résultat du test
    print("Données récupérées :", data)

    # Ajoutez des assertions en fonction de la structure des données renvoyées
    assert 10000 in data  # Vérifiez si "AMT_INCOME_TOTAL" est une clé dans le dictionnaire JSON renvoyé
    # Assurez-vous que la réponse est une liste de trois éléments
    assert isinstance(data, list)
    assert len(data) == 3


def test_infos_client():
    # URL de votre application
    url = URL_API_HOST + 'infos_client'

    # Paramètres pour simuler une requête avec un ID client (à adapter en fonction de votre application)
    params = {'id_client': '100001'}

    # Effectuer une requête GET
    response = requests.get(url, params=params)

    # Vérifier que la réponse est un succès (HTTP 200)
    assert response.status_code == 200

    # Vérifier que la réponse est au format JSON
    assert response.headers['Content-Type'] == 'application/json'

    # Analyser le contenu JSON de la réponse
    data = response.json()
    print(data)
    # Vérifier que la réponse est un succès (HTTP 200)
    assert response.status_code == 200

    # Vérifier le nombre de paramètres passés dans la requête
    # assert len(response.request.args) == len(params), "Nombre incorrect de paramètres dans la requête"
    # Vérifier le nombre de paramètres passés dans la requête
    assert len(response.url.split('?')[1].split('&')) == len(params), "Nombre incorrect de paramètres dans la requête"
    # Vérifier le nombre de valeurs retournées (remplacez le_nombre_attendu par le nombre attendu)
    le_nombre_attendu = 1  # Remplacez ceci par le nombre attendu
    assert len(data) == le_nombre_attendu, "Nombre incorrect de valeurs retournées dans la réponse"

    # # Vérifier que la réponse contient les clés attendues
    # keys_to_check = ['status_famille', 'nb_enfant', 'age', 'revenus', 'montant_credit', 'annuites', 'montant_bien']
    # for key in keys_to_check:
    #     assert key in data['0'], f"Clé manquante : {key}"
