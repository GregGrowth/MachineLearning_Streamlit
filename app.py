import streamlit as st
import pandas as pd
import io
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

st.sidebar.markdown("# Source de données 🎈")

df = pd.DataFrame()

# Options pour choisir la base de données
source_data = st.sidebar.radio(
    "Choisissez votre source de données",
    ["vin.csv", "diabete.csv", "upload file (*.csv)"]
)

if source_data == "vin.csv":
    df = pd.read_csv("./data/vin.csv", index_col=0)  # read a CSV file inside the 'data" folder next to 'app.py'
elif source_data == "diabete.csv":
    df = pd.read_csv("./data/diabete.csv", index_col=0)  # read a CSV file inside the 'data" folder next to 'app.py'
elif source_data == "upload file (*.csv)":
    # Importation du fichier CSV
    st.sidebar.header("Importez vos données")

    # Option pour modifier le séparateur et le décimal du fichier CSV
    separateur = st.sidebar.text_input("Quel est le séparateur du fichier CSV ?", ",")
    decimal = st.sidebar.text_input("Quel est le décimal du fichier CSV ?", ".")

    # Télechargement du fichier CSV
    uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

    # Lire le fichier CSV
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=separateur, decimal=decimal)

else:
    st.write("Choisissez votre source de données")

# Afficher un aperçu des données
st.subheader("Aperçu des données version originale")
st.write(df.head(5))
st.markdown("Votre base de données est de taille ...")
st.markdown("On a détecter :")


#########################################
# Étape 2 : Proposer des options de nettoyage
st.sidebar.markdown("# Nettoyage des données")

st.sidebar.markdown("## Typologie des données")
if st.sidebar.checkbox("Consulter la typologie des données"):
    st.subheader("Aperçu des informations sur la base de données")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

st.sidebar.markdown("## Imputation des données manquantes")
# Option pour supprimer les valeurs manquantes
if st.sidebar.checkbox("Supprimer les lignes avec des valeurs manquantes"):
    data = df.dropna()
    st.write("Données après suppression des valeurs manquantes :")
    st.write(data.head())

st.sidebar.markdown("## Supression des données")
# Option pour supprimer une colonne
columns = df.columns.tolist()
column_to_drop = st.sidebar.multiselect("Sélectionnez les colonnes à supprimer", columns)
if len(column_to_drop) > 0:
    data = df.drop(columns=column_to_drop)
    st.write(f"Données après suppression des colonnes {column_to_drop}:")
    st.write(df.head())

st.sidebar.markdown("## Encodage des variables catégorielles")

st.sidebar.write("Les données sont maintenant prêtes à être utilisées pour l'étape de classification.")


st.sidebar.markdown("# Playground 🎈")
type_ml = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"]
)

if type_ml == "Regression":
    regression_page()
elif type_ml == "Classification":
    classification_page()
elif type_ml == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")
    