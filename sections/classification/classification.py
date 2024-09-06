import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction principale pour la classification
def classification_page():
    # Vérifier si df est déjà chargé dans st.session_state
    if 'df' not in st.session_state:
        # Charger le fichier CSV dans df
        st.session_state['df'] = pd.read_csv("./data/vin.csv")
    df = st.session_state['df']  # Récupérer les données depuis st.session_state

    # Initialiser les clés X et y dans st.session_state si elles ne sont pas encore définies
    if 'X' not in st.session_state:
        st.session_state['X'] = None
    if 'y' not in st.session_state:
        st.session_state['y'] = None

    st.caption("Playground ML - Classification")

    # Jauge de test_size dans la barre latérale
    test_size = st.sidebar.slider("Proportion du Test (en %)", min_value=5, max_value=50, value=20, step=1, help="Choisissez une valeur comprise entre 5 et 50") / 100

    # Utilisation d'un bouton pour verrouiller/déverrouiller random_state
    if "random_state_locked" not in st.session_state:
        st.session_state.random_state_locked = True  # Par défaut, random_state est verrouillé à 42

    # Bouton pour déverrouiller le random_state
    unlock_button = st.sidebar.checkbox("Déverrouiller Random State", value=False)

    # Si le bouton est activé, permettre à l'utilisateur de saisir un random_state personnalisé
    if unlock_button:
        random_state = st.sidebar.number_input("Random State", min_value=0, max_value=100,
                                               value=st.session_state.get("custom_random_state", 42),
                                               help="Saisissez un nombre entre 0 et 100")
        st.session_state.custom_random_state = random_state  # Enregistrer la valeur personnalisée
    else:
        random_state = 42  # Par défaut, random_state est verrouillé à 42
        st.sidebar.write(f"Random State actuel : {random_state}")

    # Vérification de la valeur saisie
    if unlock_button:
        if random_state < 0 or random_state > 100:
            st.sidebar.error("Erreur : Veuillez saisir un nombre entre 0 et 100.")
        else:
            st.sidebar.write(f"Random State sélectionné : {random_state}")

    # Création des onglets principaux
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Sélection des variables", "Aperçu du dataset", "Modélisation LazyPredict",
         "Modélisation Autres", "Évaluation", "Comparaison"]
    )

    # Onglet 1 : Sélection des variables
    with tab1:
        st.subheader("Sélection des variables explicatives (X) et de la cible (y)")
        with st.expander("Description des variables"):
            st.markdown("""
            **alcohol :** Taux d'alcool dans le vin (en pourcentage).
            **malic_acid :** Quantité d'acide malique présente dans le vin, un acide organique contribuant à l'acidité du vin. Valeur continue.
            **ash :** Quantité de cendres présentes après la combustion du vin. C'est une caractéristique chimique du vin utilisée pour évaluer sa composition minérale. Valeur continue.
            **alcalinity_of_ash :** Alcalinité des cendres, une mesure du degré de neutralisation de l'acide dans le vin. Valeur continue.
            **magnesium :** Quantité de magnésium présente dans le vin. Valeur continue en milligrammes.
            **total_phenols :** Mesure de la concentration totale de phénols dans le vin. Les phénols jouent un rôle dans la couleur, l'astringence, et la stabilité du vin. Valeur continue.
            **flavanoids :** Quantité de flavonoïdes, un type spécifique de phénols. Les flavonoïdes contribuent à la couleur et au goût du vin. Valeur continue.
            **nonflavanoid_phenols :** Quantité de phénols non flavonoïdes. Valeur continue. Ces composés sont généralement présents en plus petites quantités.
            **proanthocyanins :** Mesure des proanthocyanidines, des polyphénols qui influencent la couleur et le goût du vin. Valeur continue.
            **color_intensity :** Intensité de la couleur du vin. Valeur continue.
            **hue :** Teinte du vin, qui est une mesure subjective de la couleur. Valeur continue.
            **od280/od315_of_diluted_wines :** Rapport des intensités de lumière absorbée à 280 nm et 315 nm. C'est une mesure de la qualité des protéines dans le vin. Valeur continue.
            **proline :** Quantité de proline, un acide aminé présent dans le vin, qui joue un rôle dans le goût et la structure du vin. Valeur continue.
            **target :** Classe cible du vin, ici "Vin amer". C'est la variable que nous cherchons à prédire dans notre modèle de classification.
            """)

        # Filtrer les colonnes numériques pour les variables explicatives (X)
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        # Vérifier s'il y a des colonnes numériques disponibles
        if numeric_columns:
            # Sélection des variables explicatives avec une infobulle
            selected_columns = st.multiselect(
                "Sélectionnez les variables explicatives (X)",
                options=numeric_columns,
                default=[],
                help="Les variables explicatives (X) doivent être numériques et représentent les caractéristiques ou attributs qui influencent la variable cible (y)."
            )
        else:
            # Afficher un avertissement s'il n'y a pas de colonnes numériques disponibles
            st.warning("Aucune variable numérique disponible pour les variables explicatives (X).")

        # Filtrer les colonnes catégorielles pour la cible (y)
        categorical_columns = [col for col in df.columns if
                               not pd.api.types.is_numeric_dtype(df[col]) and col not in selected_columns]

        # Ajouter une option vide au début de la liste des cibles catégorielles
        target_options = ["Choix d'une target"] + categorical_columns

        # Sélection de la cible (y) avec une infobulle
        target = st.selectbox(
            "Sélectionnez la cible (y)",
            options=target_options,
            index=0,  # Par défaut, l'option vide sera sélectionnée
            help="La cible (y) doit être une variable catégorielle que vous souhaitez prédire."
        )

        # Si des colonnes sont sélectionnées, les afficher
        if selected_columns and target != "Sélectionnez une colonne":
            # Créer trois colonnes pour simuler une ligne de séparation entre col1 et col2
            col1, col_space, col2 = st.columns([1, 0.1, 1])

            with col1:
                st.markdown("**Variables explicatives (X) sélectionnées :**")
                for column in selected_columns:
                    st.markdown(f"- {column}")

            # col_space pour gérer les espaces entre les colonnes

            with col2:
                st.markdown("**Cible (y) sélectionnée :**")
                st.markdown(f"- {target}")

            # Placer le bouton à droite
            col1, col2, col3 = st.columns([1, 1, 1])

            # Ajouter une image dans la première colonne avec l'URL


            # Bouton Valider dans la troisième colonne
            with col1:
                if st.button("Valider"):
                    # Simuler une "pop-up" avec des options de validation
                    st.write("Voulez-vous valider votre sélection ?")
                    confirm = st.radio("", ("OK", "NON"))

                    # Gérer la réponse de l'utilisateur
                    if confirm == "OK":
                        # Stocker les données si l'utilisateur valide
                        st.session_state['X'] = df[selected_columns]
                        st.session_state['y'] = df[target]
                        st.success("Votre sélection a été validée et les données ont été stockées.")
                    elif confirm == "NON":
                        st.warning("Votre sélection a été annulée. Veuillez refaire votre choix.")

            with col2:
                st.image(
                    "https://img.freepik.com/vecteurs-premium/homme-tient-symbole-point-interrogation-poser-questions-chercher-reponses-faq-concept-questions-frequemment-posees-centre-support-ligne_531064-14602.jpg",
                    caption="",
                    use_column_width=True
                )

        elif selected_columns and target == "Sélectionnez une colonne":
            st.warning("Veuillez sélectionner une colonne y.")
        else:
            st.warning("Veuillez sélectionner les colonnes pour X et y.")

    # Onglet 2 : Aperçu du dataset
    with tab2:
        # Création des sous-onglets
        subtab1, subtab2, subtab3 = st.tabs(
            ["Preview & Stats descriptives", "Matrice de corrélation", "ℹ variable du dataset"])

        # Sous-onglet 1 : Preview & Stats descriptives
        with subtab1:
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Dataset avant sélection</h5>",
                        unsafe_allow_html=True)

            # Afficher le dataset entier avec des couleurs pour les colonnes X et y
            if st.session_state.get('X') is not None and st.session_state.get('y') is not None:
                styled_data = df.style.apply(
                    lambda x: ['background-color: lightblue' if col in st.session_state['X'].columns else
                               'background-color: lightgreen' if col == st.session_state['y'].name else ''
                               for col in df.columns], axis=1
                )
                st.dataframe(styled_data)

                st.markdown(
                    """
                    <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
                        <img src="https://cdn-icons-png.flaticon.com/512/467/467262.png" width="100">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Dataset après la sélection (X et y)</h5>",
                            unsafe_allow_html=True)
                selected_data = df[st.session_state['X'].columns.tolist() + [st.session_state['y'].name]]
                styled_selected_data = selected_data.style.apply(
                    lambda x: ['background-color: lightblue' if col in st.session_state['X'].columns else
                               'background-color: lightgreen' if col == st.session_state['y'].name else ''
                               for col in selected_data.columns], axis=1
                )
                st.dataframe(styled_selected_data)

                st.markdown("---")  # Ligne de séparation

                # Résumé statistique des variables sélectionnées
                st.markdown(
                    "<h5 style='color: #FF5733; font-weight: bold;'>Résumé statistique des variables sélectionnées (X et y)</h5>",
                    unsafe_allow_html=True)
                st.write(selected_data.describe())

            else:
                st.write("Aucune sélection de variables explicatives (X) et de cible (y) n'a été effectuée.")
                st.write(df)

        # Sous-onglet 2 : Matrice de corrélation
        with subtab2:
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Matrice de corrélation</h5>",
                        unsafe_allow_html=True)

            if st.session_state.get('X') is not None:
                correlation_matrix = st.session_state['X'].corr()
                st.write(correlation_matrix)

                import seaborn as sns
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            else:
                st.warning(
                    "Veuillez sélectionner des variables explicatives (X) pour afficher la matrice de corrélation.")

        # Sous-onglet 3 : Information sur les variables
        with subtab3:
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Description des variables ℹ (X et y)</h5>",
                        unsafe_allow_html=True)

            # Description des variables, affichant les noms des colonnes avec un type et quelques stats
            if st.session_state.get('X') is not None and st.session_state.get('y') is not None:
                st.write("### Variables explicatives (X)")
                for col in st.session_state['X'].columns:
                    st.markdown(f"**{col}** : Type - {df[col].dtype}, Nombre de valeurs uniques : {df[col].nunique()}")

                st.write("### Cible (y)")
                target_col = st.session_state['y'].name
                st.markdown(
                    f"**{target_col}** : Type - {df[target_col].dtype}, Nombre de valeurs uniques : {df[target_col].nunique()}")

            else:
                st.write("Aucune sélection de variables explicatives (X) et de cible (y) n'a été effectuée.")

    # Onglet 3 : Modélisation LazyPredict
    with tab3:
        st.subheader("Modélisation rapide avec LazyPredict")

        # Vérifier que les variables X et y sont bien définies avant de continuer
        if st.session_state['X'] is not None and st.session_state['y'] is not None:

            X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'],
                                                                test_size=test_size, random_state=random_state)

            if st.button("Lancer LazyPredict"):
                lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True)
                models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
                st.write("Résultats LazyPredict :")
                st.dataframe(models)
        else:
            st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")

    # Onglet 4 : Modélisation Autres avec sous-onglets
    with tab4:
        st.subheader("Modélisation Avancée")

        # Vérifier que les variables X et y sont bien définies avant de continuer
        if st.session_state['X'] is not None and st.session_state['y'] is not None:
            subtab1, subtab2, subtab3, subtab4 = st.tabs(
                ["Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Autres"]
            )

            # Sous-onglet Logistic Regression
            with subtab1:
                st.subheader("Logistic Regression")
                max_iter = st.number_input("Max Iterations", value=100)
                model_lr = LogisticRegression(max_iter=max_iter)
                if st.button("Entraîner Logistic Regression"):
                    model_lr.fit(X_train, y_train)
                    st.write("Modèle Logistic Regression entraîné.")

            # Sous-onglet Random Forest
            with subtab2:
                st.subheader("Random Forest")
                n_estimators = st.number_input("Nombre d'arbres", value=100)
                model_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                if st.button("Entraîner Random Forest"):
                    model_rf.fit(X_train, y_train)
                    st.write("Modèle Random Forest entraîné.")

            # Sous-onglet K-Nearest Neighbors
            with subtab3:
                st.subheader("K-Nearest Neighbors")
                n_neighbors = st.number_input("Nombre de voisins", value=5)
                model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                if st.button("Entraîner K-NN"):
                    model_knn.fit(X_train, y_train)
                    st.write("Modèle K-NN entraîné.")

            # Sous-onglet Autres modèles
            with subtab4:
                st.subheader("Autres modèles")
                st.write("Sélectionnez et entraînez d'autres modèles à tester.")
        else:
            st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")

    # Onglet 5 : Évaluation
    with tab5:
        st.subheader("Évaluation des modèles")
        # Evaluation des modèles basés sur les résultats de l'entraînement
        st.write("À implémenter : Sélectionner un modèle pour l'évaluation.")

    # Onglet 6 : Comparaison (vide pour l'instant)
    with tab6:
        st.subheader("Comparaison des modèles")
        st.write("Fonctionnalité à venir.")
