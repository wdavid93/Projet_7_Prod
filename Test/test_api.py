# # test integration
import requests

# Fonction de test pour la route infos_gen
def test_site_prod_running():
    response = requests.get('https://p7.wdavid.chevaliers.oublies.fr/')
    print(response.status_code)
    assert response.status_code == 200  # Vérifiez que la réponse est HTTP 200 (OK)
