import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Classification Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Bienvenue sur le Playground de Classification")

# Étape 1 : Importation du fichier CSV
st.sidebar.header("Importez vos données")
uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"]) # indexcol=0

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

    # Autres options de nettoyage peuvent être ajoutées ici...

    st.sidebar.write("Les données sont maintenant prêtes à être utilisées pour l'étape de classification.")
else:
    st.write("Veuillez télécharger un fichier CSV pour commencer.")

