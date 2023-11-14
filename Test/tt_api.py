# # test_api.py
import sys
import os
import pytest
import json


# Ajoutez le chemin vers le répertoire contenant api.py
# Récupérer le chemin absolu du répertoire parent de votre fichier actuel (là où se trouve le fichier actuel).
current_file_directory = os.path.dirname(__file__)
print("current_file_directory : " , current_file_directory ) 
# Joindre ce chemin avec un chemin relatif pour atteindre le répertoire "api_docker" et obtenir un chemin absolu.
api_docker_directory = os.path.abspath(os.path.join(current_file_directory, "..", "api_docker"))
print("api_docker_directory : " , api_docker_directory ) 
# Insérer ce chemin au début du chemin de recherche Python (sys.path) pour que Python puisse trouver les modules dans ce répertoire.
sys.path.insert(0, api_docker_directory)
# Importez votre module API
from API import API

# current_file_directory = os.path.dirname(__file__)
# print("update current_file_directory : " , current_file_directory ) 
# os.chdir(api_docker_directory)

# # def test_home_page(client):
# #     response = init_model()
# #     assert "Initialisation terminée.".encode('utf-8') in response.data
# def test_infos_client_route(client):
#     response = client.get("/infos_client")
#     assert response.status_code == 200

# if __name__ == "__main__":
#     pytest.main()
# import pytest
# from API import API  # Assurez-vous d'importer l'application Flask de votre application principale


# Fixture pour initialiser l'application pour les tests
@pytest.fixture
def client():
    client = app.test_client()
    yield client

# Fonction de test pour la route infos_gen
def test_infos_gen_route(client):
    response = client.get("/infos_gen")  # Envoyer une requête GET à la route
    assert response.status_code == 200  # Vérifier que la réponse est HTTP 200 (OK)
    
    # Vérifier le contenu de la réponse (au format JSON)
    data = json.loads(response.get_data(as_text=True))
    
    # Vérifiez si la réponse contient des valeurs spécifiques, par exemple :
    assert "AMT_INCOME_TOTAL" in data  # Vérifiez si "AMT_INCOME_TOTAL" est une clé dans le dictionnaire JSON renvoyé

    # Vous pouvez ajouter d'autres vérifications en fonction de la structure des données renvoyées par la route


