# sourceData_page.py
import streamlit as st
import pandas as pd

# Modifier la signature de la fonction pour accepter df
def sourceData_page(df):
    st.subheader("Source de données")

    # Exemple d'interaction avec df
    if st.checkbox("Afficher les premières lignes du fichier de données"):
        st.write(df.head())

    # Option d'importation de nouveaux fichiers par l'utilisateur
    uploaded_file = st.file_uploader("Télécharger un fichier CSV", type="csv")

    # Si l'utilisateur charge un nouveau fichier, remplacer le DataFrame existant
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Nouveau fichier chargé avec succès !")
        st.write(df.head())

    # Retourner le DataFrame (modifié ou non)
    return df
