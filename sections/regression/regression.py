import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def regression_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans le Playground de Regression")
    st.title("Dataframe Diabete")

    # Chargement des données depuis le local
    df = pd.read_csv("data/diabete.csv",index_col=0)
    st.write(df)

    # Preparation des données
    X = df.drop(columns=['target'])
    y = df['target']

    # Affichage des données
    st.subheader("Données")
    st.write(X)
    st.subheader("Target")
    st.write(y)

    # Separation des données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modele régréssion linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    comparaison = pd.DataFrame({
        'Valeur Réelle (y_test)': y_test,
        'Valeur Prédite (y_pred)': y_pred
    })


    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Afficher les résultats dans Streamlit
    st.subheader("Prédictions")
    st.dataframe(y_pred)

    st.subheader("Evaluation")
    st.write(mse)
    st.write(r2)

    st.subheader("Comparaison entre y_test et y_pred")
    st.dataframe(comparaison)

    #############################################
    #Lazypredict
    from lazypredict.Supervised import LazyRegressor

    reg = LazyRegressor()
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    st.subheader("Comparaison des Modèles")
    st.dataframe(models)

