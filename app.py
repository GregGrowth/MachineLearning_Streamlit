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

# Logo diginamic
st.logo("https://studl.com/assets/uploads/evenement/image6058c34c4b8a77.08689920.png")

# Barre horizontale en haut
st.markdown("""
    <style>
    .top-bar {
        background-color: #F0F2F6;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: right;
    }
    .top-bar a {
        margin: 0 30px;
        text-decoration: none;
        color: #F90100;
    }
    .top-bar a:hover {
        color: #0056b3;
    }
    </style>
    <div class="top-bar">
        <a href="#readme" target="_self">README</a> |
        <a href="https://github.com/mkunegel/ProjetML" target="_blank">Lien GitHub du projet ML</a>
    </div>
    """, unsafe_allow_html=True)

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
