# Importation des bibliothèques
import streamlit as st
# from stcomponents import st_tab

import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap
# import ipython
from PIL import Image
from flask import Flask
import streamlit.components.v1 as components
# from evidently.dashboard import Dashboard
# from evidently.tabs import DataDriftTab

# from "C:\Users\Zbook\OpenClassRoom\ProjetsGitsOCR\Projet_7\api_docker\api" import get_shap_values, predict_explanation


# Définition de l'URL de l'API
# URL_API = "http://localhost:5001/"  # Utilisation en local
URL_API = "http://projet7api:5001/"  # Utilisation en production


def main():
    # # Code HTML pour un bouton de zoom
    # zoom_button = """
    # <button onclick="zoomIn()">Zoom In</button>
    # <button onclick="zoomOut()">Zoom Out</button>
    # """

    # # Code JavaScript pour effectuer le zoom
    # javascript = """
    # <script>
    # function zoomIn() {
    # document.body.style.zoom = "150%";
    # }

    # function zoomOut() {
    # document.body.style.zoom = "100%";
    # }
    # </script>
    # """

    # # Afficher le bouton et le code JavaScript
    # st.markdown(zoom_button, unsafe_allow_html=True)
    # st.markdown(javascript, unsafe_allow_html=True)

    # st.sidebar.title("Navigation")
    # selected_tab = st.sidebar.radio("Sélectionnez une option :", ["Onglet 1", "Onglet 2"])
    # selected_tab = st.tabs(["Client", "Shap","Clients"])
    # Affichez le nom de l'onglet sélectionné

    # Créez les onglets
    tab1, tab2, tab3 = st.tabs(["Client", "Shap", "Clients"])

    # Utilisez le libellé pour prendre des décisions
    with tab1:
        st.write("Vous avez sélectionné l'onglet Client.")
        # st.text(f"Vous avez sélectionné l'onglet : {selected_tab.label}")

        # Initialisation de l'application

        init = st.markdown("*Initialisation de l'application en cours...*")
        init = st.markdown(init_api())
        # st.write("Version mlflow: ", mlflow_version())
        # Affichage du titre et du sous-titre
        st.title("Implémenter un modèle de scoring")
        st.markdown("<i>API répondant aux besoins du projet 7 pour le parcours Data Scientist OpenClassRoom</i>",
                    unsafe_allow_html=True)

        # Affichage d'informations dans la sidebar
        st.sidebar.subheader("Informations générales")

        # Chargement du logo
        logo = load_logo()
        st.sidebar.image(logo, width=200)

        # Chargement de la selectbox
        lst_id = load_selectbox()
        global id_client
        id_client = st.sidebar.selectbox("ID Client", lst_id)

        # Chargement des infos générales
        nb_credits, rev_moy, credits_moy, targets = load_infos_gen()

        # Affichage des infos dans la sidebar
        st.sidebar.markdown(
            "<u>Nombre crédits existants dans la base :</u>", unsafe_allow_html=True)
        st.sidebar.text(nb_credits)

        # # Graphique camembert
        # st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)
        # plt.pie(targets, explode=[0, 0.1], labels=["Solvable", "Non solvable"], autopct='%1.1f%%', shadow=True, startangle=90)
        # st.sidebar.pyplot()
        # Graphique camembert
        st.sidebar.markdown(
            "<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)
        # Désactivez l'avertissement lié à Matplotlib
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots()
        ax.pie(targets, explode=[0, 0.1], labels=[
            "Solvable", "Non solvable"], autopct='%1.1f%%', shadow=True, startangle=90)
        st.sidebar.pyplot(fig)

        # Revenus moyens
        st.sidebar.markdown("<u>Revenus moyens $(USD) :</u>",
                            unsafe_allow_html=True)
        st.sidebar.text(rev_moy)

        # Montant crédits moyen
        st.sidebar.markdown(
            "<u>Montant crédits moyen $(USD) :</u>", unsafe_allow_html=True)
        st.sidebar.text(credits_moy)

        # if selected_tab == "Client":
        # Affichage de l'ID client sélectionné
        st.write("Vous avez sélectionné le client :", id_client)

        # Affichage état civil
        st.header("**Informations client**")
        infos_client = identite_client()
        if st.checkbox("Afficher les informations du client?"):
            infos_client = identite_client()
            st.write("Statut famille :**",
                    infos_client["NAME_FAMILY_STATUS"][0], "**")
            st.write("Nombre d'enfant(s) :**",
                    infos_client["CNT_CHILDREN"][0], "**")
            st.write("Age client :", int(
                infos_client["DAYS_BIRTH"].values / -365), "ans.")

            data_age = load_age_population()
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(9, 9))
            plt.hist(data_age, edgecolor='k', bins=25)
            plt.axvline(
                int(infos_client["DAYS_BIRTH"].values / -365), color="red", linestyle=":")
            plt.title('Age of Client')
            plt.xlabel('Age (years)')
            plt.ylabel('Count')
            # Avant d'afficher les graphiques, désactivez l'avertissement
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.pyplot()

            # st.subheader("*Revenus*")
            # st.write("Total revenus client :", infos_client["AMT_INCOME_TOTAL"][0], "$")

            # data_revenus = load_revenus_population()
            # plt.style.use('fivethirtyeight')
            # plt.figure(figsize=(9, 9))
            # plt.hist(data_revenus, edgecolor='k')
            # plt.axvline(infos_client["AMT_INCOME_TOTAL"][0], color="red", linestyle=":")
            # plt.title('Revenus du Client')
            # plt.xlabel('Revenus ($ USD)')
            # plt.ylabel('Count')
            # # Avant d'afficher les graphiques, désactivez l'avertissement
            # st.set_option('deprecation.showPyplotGlobalUse', False)

            # st.pyplot()
            st.subheader("*Revenus*")
            st.write("Total revenus client :",
                    infos_client["AMT_INCOME_TOTAL"][0], "$")

            data_revenus = load_revenus_population()

            # Set the style of plots
            plt.style.use('fivethirtyeight')
            fig, ax = plt.subplots(figsize=(9, 9))

            # Plot the distribution of revenus
            ax.hist(data_revenus, edgecolor='k')
            ax.axvline(infos_client["AMT_INCOME_TOTAL"]
                    [0], color="red", linestyle=":")
            ax.set_title('Revenus du Client')
            ax.set_xlabel('Revenus ($ USD)')
            ax.set_ylabel('Count')

            # Avant d'afficher les graphiques, désactivez l'avertissement
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.pyplot(fig)

            #

            st.write("Montant du crédit :", infos_client["AMT_CREDIT"][0], "$")
            st.write("Annuités crédit :", infos_client["AMT_ANNUITY"][0], "$")
            st.write("Montant du bien pour le crédit :",
                    infos_client["AMT_GOODS_PRICE"][0], "$")
        else:
            st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)

        # Affichage solvabilité client
        st.header("**Analyse dossier client**")
        st.markdown("<u>Probabilité de risque de faillite du client :</u>",
                    unsafe_allow_html=True)
        prediction = load_prediction(id_client)
        st.write(round(prediction * 100, 2), "%")
        st.markdown("<u>Données client :</u>", unsafe_allow_html=True)
        st.write(identite_client())
    #
        # Affichage des dossiers similaires
        chk_voisins = st.checkbox("Afficher dossiers similaires?")

        if chk_voisins:
            similar_id = load_voisins()
            st.markdown(
                "<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
            st.write(similar_id)
            st.markdown("<i>Target 1 = Client en faillite</i>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)
    with tab2:
        st.write("Vous avez sélectionné l'onglet Shap.")
        
        st.header("**Information des Shap**")
        # if st.checkbox("Afficher les informations SHAP?"):
        # Appel de la fonction pour obtenir les valeurs SHAP
        # Appel à la fonction pour afficher les graphiques SHAP
        shap.initjs()
        option = st.selectbox("Sélectionnez une option", ["Shap 1", "Shap 2", "Shap 3", "Shap 4"])
        # st.write("SHAP 1 id_client", id_client)

        X_json = requests.get(URL_API + "shap_xgb_x")
        expected_json = requests.get(URL_API + "shap_xgb_expected")
        values_json = requests.get(URL_API + "shap_xgb_values")
        df_json = requests.get(URL_API + "shap_xgb_df")

        X = np.asarray(X_json.json())
        expected = expected_json.json()
        values = np.asarray(values_json.json())
        df = pd.DataFrame.from_dict(df_json.json())
        # st.write("SHAP 1 df", df)
        if option == "Shap 1":
            st_shap(shap.force_plot(expected, values[0, :], df.iloc[0, :]))
        if option == "Shap 2":    
            st_shap(shap.force_plot(expected, values[:500, :], df.iloc[:500, :]), 500)
        if option == "Shap 3":
            plt.figure(figsize=[10,10])
            shap.summary_plot(values, df, plot_type="bar", show=False)
            plt.savefig("shap_summary_plot_bar.png")
            plt.show()
            image = Image.open('shap_summary_plot_bar.png')
            st.image(image)
        if option == "Shap 4":
            plt.figure(figsize=[10,10])
            shap.summary_plot(values, X, show=False)
            plt.savefig("shap_summary_plot_values.png")
            plt.show()
            image = Image.open('shap_summary_plot_values.png')
            st.image(image)
    with tab3:
        st.write("Vous avez sélectionné l'onglet Clients.")       
        st.header("**Information des clients**")
        # Sélectionnez une option pour explorer les données
        option = st.selectbox("Sélectionnez une option", ["Âge", "Revenus"])
        df_json = requests.get(URL_API + "shap_xgb_df")
        df = pd.DataFrame.from_dict(df_json.json())
        if option == "Âge":
            st.subheader("Âge des Clients")
            fig = px.histogram(df, x=df["DAYS_BIRTH"] / -365, nbins=25)
            st.plotly_chart(fig)
            tab3.line_chart(fig)

        elif option == "Revenus":
            # st.subheader("Revenus des Clients")
            # fig = px.histogram(data_train, x="revenus")
            # fig = px.histogram(df, x="AMT_INCOME_TOTAL")
            # Définir la plage de l'axe horizontal entre 0 et 20
            # fig.update_xaxes(range=[0, 5])
            # Filtrer le DataFrame pour ne conserver que les revenus entre 0 et 50000
            filtered_df = df[(df['AMT_INCOME_TOTAL'] >= 0) & (df['AMT_INCOME_TOTAL'] <= 100000)]

            st.subheader("Revenus des Clients")
            fig = px.histogram(filtered_df, x="AMT_INCOME_TOTAL")
            st.plotly_chart(fig)

    # df_json_train = requests.get(URL_API + "shap_xgb_df")
    # data_train_e = pd.DataFrame.from_dict(df_json_train.json())    
    # df_test = requests.get(URL_API + "load_df_test")
    # data_test_e = pd.DataFrame.from_dict(df_test.json())
    # # Créer un DataFrame contenant une colonne pour identifier les données d'entraînement et de test
    # data_train_e['dataset'] = 'train'
    # data_test_e['dataset'] = 'test'

    # # Concaténer les données d'entraînement et de test
    # df = pd.concat([data_train_e, data_test_e])

    # # Créer un DataFrame de référence (données d'entraînement)
    # df_reference = data_train_e

    # # Créer une instance de la classe DataDriftTab
    # data_drift_tab = DataDriftTab(df_reference, df, task='classification')

    # # Créer un tableau de bord Evidently avec le tab Data Drift
    # dashboard = Dashboard(tabs=[data_drift_tab])

    # # Générer le rapport Evidently
    # report = dashboard.run()

    # # Afficher le rapport Evidently
    # report.show()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Fonction pour l'initialisation de l'API
@st.cache_data
def init_api():
    init_api = requests.get(URL_API + "init_model")
    init_api = init_api.json()
    return "Initialisation application terminée."

# Fonction pour charger le logo


@st.cache_data
def load_logo():
    logo = Image.open("logo.png")
    return logo

# Fonction pour charger la liste des ID clients


@st.cache_data
def load_selectbox():
    data_json = requests.get(URL_API + "load_data")
    data = data_json.json()
    lst_id = [i[0] for i in data]
    return lst_id

# Fonction pour charger les informations générales


@st.cache_data
def load_infos_gen():
    infos_gen = requests.get(URL_API + "infos_gen")
    infos_gen = infos_gen.json()
    nb_credits = infos_gen[0]
    rev_moy = infos_gen[1]
    credits_moy = infos_gen[2]
    targets = requests.get(URL_API + "disparite_target")
    targets = targets.json()
    return nb_credits, rev_moy, credits_moy, targets

# Fonction pour récupérer les informations du client sélectionné


@st.cache_data
def identite_client():
    infos_client = requests.get(
        URL_API + "infos_client", params={"id_client": id_client})
    infos_client = json.loads(infos_client.content.decode("utf-8"))
    infos_client = pd.DataFrame.from_dict(infos_client).T
    return infos_client

# Fonction pour charger l'âge de la population


@st.cache_data
def load_age_population():
    data_age_json = requests.get(URL_API + "load_age_population")
    data_age = data_age_json.json()
    return data_age


@st.cache_data  # 👈 Add the caching decorator
def load_revenus_population():
    # Requête permettant de récupérer des tranches de revenus
    # de la population pour le graphique situant le client
    data_revenus_json = requests.get(URL_API + "load_revenus_population")

    data_revenus = data_revenus_json.json()

    return data_revenus

# @st.cache_data
def predict_explanation():
    try:
        explanation_data = requests.get(URL_API + "predict_explanation")
        st.write(f"explanation_data  : {explanation_data}")
        explanation_data = explanation_data.json()

        # Si vous avez obtenu des données SHAP, vous pouvez les afficher.
        if "data" in explanation_data:
            shap_df = explanation_data["data"]
            shap_values = shap_df['data']
            feature_names = shap_df['columns']

            # Affichez le graphique SHAP
            shap.summary_plot(shap_values, feature_names=feature_names)
        else:
            st.error("Les données SHAP ne sont pas disponibles dans la réponse.")
    except json.JSONDecodeError as e:
        st.error(f"Erreur lors de la récupération des explications : {e}")


def generate_decision_plot(log_reg_explainer, X_test, row_index=0, highlight=None):
    # Convertit la valeur attendue en liste
    expected_value = [log_reg_explainer.expected_value.tolist()]
    # Convertit les valeurs SHAP en liste
    shap_values = [log_reg_explainer.shap_values(X_test)]
    # Convertit les noms de colonnes en liste
    feature_names = df_train.columns.tolist()

    shap.multioutput_decision_plot(
        expected_value, shap_values, row_index=row_index, feature_names=feature_names, highlight=highlight)


def generate_summary_plot(log_reg_explainer, X_test):
    shap.summary_plot(log_reg_explainer.shap_values(X_test),
                      feature_names=df_train.columns)
@st.cache_data
def load_prediction(id_client):
    try:
        prediction = requests.get(
            URL_API + "predict", params={"id_client": id_client})
        prediction = prediction.json()
        if prediction:
            return prediction[1]
    except ValueError:
        st.error("Erreur lors de la récupération de la prédiction.")
    return 0

@st.cache_data  # Utilisation de la fonction de cache de Streamlit pour éviter des appels répétés à l'API
def load_voisins():
    try:
        # Envoie une requête à l'API pour récupérer les dossiers similaires au client sélectionné
        voisins = requests.get(URL_API + "load_voisins",
                               params={"id_client": id_client})

        # Transforme la réponse en dictionnaire Python
        voisins = json.loads(voisins.content.decode("utf-8"))

        # Transforme le dictionnaire en un DataFrame pour une meilleure manipulation
        voisins = pd.DataFrame.from_dict(voisins).T

        # Extraction de la colonne "TARGET" (solvabilité) pour plus de lisibilité
        target = voisins["TARGET"]
        # Suppression de la colonne "TARGET" du DataFrame (afin d'éviter les redondances)
        voisins.drop(labels=["TARGET"], axis=1, inplace=True)
        # Réinsère la colonne "TARGET" en première position dans le DataFrame
        voisins.insert(0, "TARGET", target)

        # Renvoie le DataFrame contenant les dossiers similaires
        return voisins
    except ValueError:
        st.error("Erreur lors de la récupération des voisins.")


if __name__ == "__main__":
    main()
