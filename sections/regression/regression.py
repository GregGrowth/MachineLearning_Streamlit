import streamlit as st
from sections.regression.apercu import show_apercu
from sections.regression.modeles import show_modeles
from sections.regression.visuels import show_visuels
from sections.regression.comparaison import show_comparaison

# Fonction principale pour le Playground de Régression
def regression_page():
    st.caption("Bienvenue dans le Playground de Régression")

    # Sidebar pour les paramètres de base
    test_size = st.sidebar.slider("Proportion du Test (en %)", min_value=5, max_value=50, value=20, step=1) / 100
    random_state = st.sidebar.number_input("Random State", value=42)

    # Onglets principaux : Aperçu des données, Modèles, Visuels, Comparaison des modèles
    apercu_tab, modeles_tab, visuels_tab,comparaison_tab = st.tabs(["Aperçu des Données", "Modèles", "Visuels","Comparaison des modèles"])

    # Appel des fonctions pour chaque onglet
    with apercu_tab:
        X_train, X_test, y_train, y_test = show_apercu(test_size, random_state)

    with modeles_tab:
        show_modeles(X_train, X_test, y_train, y_test, random_state)

    with visuels_tab:
        show_visuels(X_train, X_test, y_train, y_test)

    with comparaison_tab:
        show_comparaison(X_train, X_test, y_train, y_test)
