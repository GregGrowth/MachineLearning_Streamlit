import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def classification_page():
    st.header("Bienvenue dans le Playground de Classification")

    # Charger le fichier vin.csv
    @st.cache
    def load_data():
        return pd.read_csv("data/vin.csv")

    data = load_data()

    st.subheader("Aperçu du jeu de données")
    st.write(data.head())

    # Séparer les variables explicatives (X) et la variable cible (y)
    X = data.drop(columns=['target', 'Unnamed: 0'])
    y = data['target']

    # Choix de l'utilisateur pour la taille du split
    test_size = st.sidebar.slider("Taille du jeu de test (%)", 10, 50, 20) / 100
    random_state = st.sidebar.number_input("Random State", value=42)

    # Split du jeu de données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.subheader("Taille des ensembles d'entraînement et de test")
    st.write(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
    st.write(f"Taille de l'ensemble de test : {X_test.shape[0]}")

    # Instanciation du modèle et entraînement
    st.subheader("Entraînement du modèle")
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    st.write("Modèle entraîné avec succès.")

    # Prédiction
    st.subheader("Prédiction")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    st.write("Prédictions sur l'ensemble de test :")
    st.write(y_pred)

    st.write("Probabilités de prédiction :")
    st.write(y_prob)

    # Évaluation
    st.subheader("Évaluation du modèle")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    st.write("Matrice de confusion :")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Rapport de classification
    cr = classification_report(y_test, y_pred, output_dict=True)
    st.write("Rapport de classification :")
    st.json(cr)
