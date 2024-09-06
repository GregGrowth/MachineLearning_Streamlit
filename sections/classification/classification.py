import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def classification_page():

    st.header("Bienvenue")
    st.caption("Bienvenue dans le Playground de classification")

    # Charger le fichier vin.csv
    @st.cache_data
    def load_data():
        return pd.read_csv("data/vin.csv")

    data = load_data()
    # data = st.session_state['df']

    st.subheader("Aperçu du jeu de données")
    st.write(data.head())

    # Séparer les variables explicatives (X) et la variable cible (y)
    X = data.drop(columns=['target', 'Unnamed: 0'])
    y = data['target']

    # Sélection des variables explicatives par l'utilisateur
    st.sidebar.markdown("## Sélection des variables explicatives")
    selected_columns = st.sidebar.multiselect("Sélectionnez les variables explicatives", options=X.columns.tolist(), default=X.columns.tolist())

    if selected_columns:
        X = X[selected_columns]
    else:
        st.warning("Aucune colonne sélectionnée pour l'entraînement.")



    # Choix de l'utilisateur pour la taille du split
    test_size = st.sidebar.slider("Taille du jeu de test (%)", 10, 50, 20) / 100
    random_state = st.sidebar.number_input("Random State", value=42)

    # Split du jeu de données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.subheader("Taille des ensembles d'entraînement et de test")
    st.write(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
    st.write(f"Taille de l'ensemble de test : {X_test.shape[0]}")



    # Sélection du modèle par l'utilisateur
    model_choice = st.sidebar.selectbox("Choisissez un modèle",
                                        ("Logistic Regression",
                                         "K-Nearest Neighbors",
                                         "Random Forest"))

    # Instanciation du modèle choisi
    if model_choice == "Logistic Regression":
        clf = LogisticRegression(max_iter=200)
    elif model_choice == "K-Nearest Neighbors":
        clf = KNeighborsClassifier()
    elif model_choice == "Random Forest":
        clf = RandomForestClassifier()

    # Option pour optimiser les hyperparamètres avec GridSearchCV
    if st.sidebar.checkbox("Optimiser les hyperparamètres (GridSearch)"):
        if model_choice == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
            grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, refit=True, verbose=0)
            grid.fit(X_train, y_train)
            st.write(f"Meilleurs hyperparamètres : {grid.best_params_}")
            clf = grid.best_estimator_

    # Entraîner le modèle
    st.subheader(f"Entraînement du modèle : {model_choice}")
    clf.fit(X_train, y_train)
    st.write(f"{model_choice} entraîné avec succès.")

    # Prédiction
    st.subheader("Prédiction")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    st.write("Prédictions sur l'ensemble de test :")
    st.write(y_pred)

    st.write("Probabilités de prédiction :")
    st.write(y_prob)

    # Évaluation du modèle
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

    # Afficher les coefficients du modèle si Logistic Regression est sélectionné
    if model_choice == "Logistic Regression":
        st.subheader("Coefficients du modèle")
        coefficients = pd.DataFrame(clf.coef_[0], X.columns, columns=['Coefficient'])
        st.write(coefficients)

    # Courbe ROC et AUC
    st.subheader("Courbe ROC")
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlabel('Taux de Faux Positifs')
    ax.set_ylabel('Taux de Vrais Positifs')
    ax.set_title('Courbe ROC')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Validation croisée
    st.subheader("Validation croisée")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    st.write(f"Scores de validation croisée : {cv_scores}")
    st.write(f"Moyenne des scores : {cv_scores.mean()}")
