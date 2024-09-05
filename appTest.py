
# Autre code

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(
    page_title="Exploration des données",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Exploration des données")

# Étape 1 : Importer un fichier CSV
uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lire les données
    data = pd.read_csv(uploaded_file)

    # Affichage d'un aperçu des données
    st.subheader("Aperçu des données")
    st.write(data.head())

    # Afficher les informations générales
    st.subheader("Informations générales")
    st.write(data.info())

    # Résumé statistique
    st.subheader("Résumé statistique")
    st.write(data.describe())

    # Distribution des colonnes numériques
    st.subheader("Distribution des variables numériques")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    selected_column = st.selectbox("Sélectionnez une colonne pour afficher sa distribution", numeric_columns)

    fig, ax = plt.subplots()
    sns.histplot(data[selected_column], kde=True, ax=ax)
    ax.set_title(f"Distribution de {selected_column}")
    st.pyplot(fig)

    # Matrice de corrélation
    st.subheader("Matrice de corrélation")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

else:
    st.write("Veuillez télécharger un fichier CSV pour commencer.")

    # Générer un rapport Sweetviz
    if st.button("Générer un rapport Sweetviz"):
        report = sv.analyze(data)
        report.show_html(filepath='sweetviz_report.html', open_browser=False)
        st.success("Rapport Sweetviz généré ! Veuillez consulter le fichier sweetviz_report.html")
