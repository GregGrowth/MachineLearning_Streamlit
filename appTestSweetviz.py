import streamlit as st
import pandas as pd
import sweetviz as sv
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page

# Configuration de la page principale
st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Choix du playground dans la sidebar
type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"]
)

if type_data == "Regression":
    regression_page()

elif type_data == "Classification":
    st.title("Bienvenue sur le Playground de Classification")

    # Étape 1 : Importation du fichier CSV
    st.sidebar.header("Importez vos données")
    uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

    if uploaded_file is not None:
        # Lire le fichier CSV
        data = pd.read_csv(uploaded_file)

        # Afficher un aperçu des données
        st.subheader("Aperçu des données")
        st.write(data.head())

        # Étape 2 : Proposer des options de nettoyage
        st.sidebar.subheader("Options de nettoyage des données")

        # Option pour supprimer les valeurs manquantes
        if st.sidebar.checkbox("Supprimer les lignes avec des valeurs manquantes"):
            data = data.dropna()
            st.write("Données après suppression des valeurs manquantes :")
            st.write(data.head())

        # Option pour supprimer une colonne
        columns = data.columns.tolist()
        column_to_drop = st.sidebar.multiselect("Sélectionnez les colonnes à supprimer", columns)
        if len(column_to_drop) > 0:
            data = data.drop(columns=column_to_drop)
            st.write(f"Données après suppression des colonnes {column_to_drop}:")
            st.write(data.head())

        # Ajouter l'option de génération du rapport Sweetviz
        st.sidebar.subheader("Générer un rapport Sweetviz")
        if st.sidebar.button("Générer le rapport Sweetviz"):
            with st.spinner('Génération du rapport Sweetviz...'):
                report = sv.analyze(data)
                report.show_html(filepath='sweetviz_report.html', open_browser=False)
            st.success("Rapport Sweetviz généré !")

            # Offrir un lien de téléchargement du fichier HTML généré
            with open("sweetviz_report.html", "rb") as file:
                st.download_button(
                    label="Télécharger le rapport Sweetviz",
                    data=file,
                    file_name="sweetviz_report.html",
                    mime="text/html"
                )

        st.sidebar.write("Les données sont maintenant prêtes à être utilisées pour l'étape de classification.")
    else:
        st.write("Veuillez télécharger un fichier CSV pour commencer.")

elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")
