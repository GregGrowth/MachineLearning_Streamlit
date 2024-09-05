import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le fichier classification.py
from sections.classification.classification import classification_page

st.set_page_config(
    page_title="Playground de Classification",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Classification Playground")

# Charger le fichier vin.csv
@st.cache_data
def load_data():
    return pd.read_csv("./data/vin.csv")

data = load_data()

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
                                    ("Logistic Regression",))

# Instanciation du modèle choisi
if model_choice == "Logistic Regression":
    clf = LogisticRegression(max_iter=200)

# Option pour optimiser les hyperparamètres avec GridSearchCV
if st.sidebar.checkbox("Optimiser les hyperparamètres (GridSearch)"):
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
sns.heatmap(cm, annot=True, fmt
