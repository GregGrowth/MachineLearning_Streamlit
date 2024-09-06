import streamlit as st
from sections.dataImport.dataImport import sourceData_page
from sections.dataPreprocessing.dataPreprocessing import nettoyageData_page
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page

st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("# Projet Machine Learning 🎈")
st.sidebar.markdown("# Main page 🎈")

# Barre de navigation principale
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choisissez une page",
    ["Accueil", "Source de données", "Nettoyage des données", "Playground"]
)

# Gestion des pages
if page == "Accueil":
    st.write("Bienvenue sur l'application Machine Learning Playground!")
    st.write("Utilisez la barre latérale pour naviguer vers les différentes sections.")
elif page == "Source de données":
    sourceData_page()  # Appel de la page pour l'import des données
elif page == "Nettoyage des données":
    nettoyageData_page()  # Appel de la page pour le nettoyage des données
elif page == "Playground":

    type_ml = st.sidebar.radio(
        "Choisissez votre type de playground",
        ["Regression", "Classification", "NailsDetection"],
        index=None
    )

    if type_ml == "Regression":
        regression_page()
    elif type_ml == "Classification":
        classification_page()
    elif type_ml == "NailsDetection":
        nail_page()
    else:
        st.write("Choisissez une option")

# app.py, run with 'streamlit run app.py'