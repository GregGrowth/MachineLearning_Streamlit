import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lazypredict.Supervised import LazyClassifier
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


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
    test_size = st.sidebar.slider("Proportion du Test (en %)", min_value=5, max_value=50, value=20, step=1,
                                  help="Choisissez une valeur comprise entre 5 et 50") / 100

    # Taille totale de la population basée sur le DataFrame sélectionné
    total_population = df.shape[0]

    # Calcul des populations pour le train et le test
    test_population = int(total_population * test_size)
    train_population = total_population - test_population

    # Encadrer les résultats avec moins d'espacement entre les lignes
    st.sidebar.markdown(f"""
        <div style="border: 1px solid #aaa; padding: 10px; border-radius: 5px;">
            <p style="margin: 0;text-align: center;">Population test : {test_population}</p>
            <p style="margin: 0;text-align: center;">Population train : {train_population}</p>
            <p style="margin: 0; text-align: center;">Taille totale de la population : {total_population}</p>
        </div>
        """, unsafe_allow_html=True)

    # Jauge de random_state dans la barre latérales
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Sélection des variables", "Aperçu du dataset", "Modélisation", "Évaluation", "Comparaison"]
    )

    # Onglet 1 : Sélection des variables
    with tab1:
        st.markdown("<h6 style='margin-top: 20px;'> ⓪ Voici quelques explications sur le dataset du 'vin' </h6>",
                    unsafe_allow_html=True)

        # Créer un DataFrame avec les descriptions de variables
        data_description_vin = {
            "Nom de la variable": [
                "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols",
                "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity",
                "hue", "od280/od315_of_diluted_wines", "proline", "target"
            ],
            "Description": [
                "Taux d'alcool dans le vin (en pourcentage).",
                "Quantité d'acide malique présente dans le vin, un acide organique contribuant à l'acidité du vin. Valeur continue.",
                "Quantité de cendres présentes après la combustion du vin. C'est une caractéristique chimique du vin utilisée pour évaluer sa composition minérale. Valeur continue.",
                "Alcalinité des cendres, une mesure du degré de neutralisation de l'acide dans le vin. Valeur continue.",
                "Quantité de magnésium présente dans le vin. Valeur continue en milligrammes.",
                "Mesure de la concentration totale de phénols dans le vin. Les phénols jouent un rôle dans la couleur, l'astringence, et la stabilité du vin. Valeur continue.",
                "Quantité de flavonoïdes, un type spécifique de phénols. Les flavonoïdes contribuent à la couleur et au goût du vin. Valeur continue.",
                "Quantité de phénols non flavonoïdes. Valeur continue. Ces composés sont généralement présents en plus petites quantités.",
                "Mesure des proanthocyanidines, des polyphénols qui influencent la couleur et le goût du vin. Valeur continue.",
                "Intensité de la couleur du vin. Valeur continue.",
                "Teinte du vin, qui est une mesure subjective de la couleur. Valeur continue.",
                "Rapport des intensités de lumière absorbée à 280 nm et 315 nm. C'est une mesure de la qualité des protéines dans le vin. Valeur continue.",
                "Quantité de proline, un acide aminé présent dans le vin, qui joue un rôle dans le goût et la structure du vin. Valeur continue.",
                "Classe cible du vin, ici 'Vin amer'. C'est la variable que nous cherchons à prédire dans notre modèle de classification."
            ]
        }

        # Créer un DataFrame à partir du dictionnaire
        df_description_vin = pd.DataFrame(data_description_vin)

        # Afficher le tableau dans l'interface Streamlit
        with st.expander("Description des variables du dataset Vin"):
            st.dataframe(df_description_vin)

        st.markdown("---")

        st.markdown("<h6>➀ Vous pouvez sélectionner vos variables explicatives (X). Pour information, votre variable 'target' est déjà définie !</h6>", unsafe_allow_html=True)

        # Sélectionner une colonne 'target' comme cible (y)
        if 'target' not in df.columns:
            st.error("La colonne 'target' n'existe pas dans le dataset.")
        else:
            # Définir la colonne target comme variable cible (y)
            target = 'target'
            st.write(f"La variable cible (y) est : {target}")

            # Filtrer les colonnes numériques pour les variables explicatives (X) en excluant 'target'
            numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != target]

            # Vérifier s'il y a des colonnes numériques disponibles
            if numeric_columns:
                # Sélection des variables explicatives avec une infobulle
                selected_columns = st.multiselect(
                    "Sélectionnez les variables explicatives (X)",
                    options=numeric_columns,
                    default=[],
                    help="Les variables explicatives (X) doivent être numériques et influencent la variable cible (y)."
                )
            else:
                st.warning("Aucune variable numérique disponible pour les variables explicatives (X).")

        st.markdown("---")
        st.markdown("<h6>➁ Voici un aperçu sous forme de tableau de vos données sélectionnées ! Si vous êtes sure de votre choix, appuyer sur le bouton 'Valider' en bas de cette page.</h6>", unsafe_allow_html=True)

        # Afficher un message d'avertissement si aucune colonne n'est sélectionnée
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

            # Ajouter une ligne horizontale
            st.markdown("---")

            # Placer le bouton à droite
            col1, col2, col3 = st.columns([1, 1, 1])

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
            ["Preview & Stats descriptives", "Matrice de corrélation & Pairplot", "ℹ variable du dataset"])

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

                st.markdown("---")  # Ligne de séparation

                # Visualisation des distributions des variables explicatives (X)
                st.markdown(
                    "<h6 style='color: #000000; font-weight: bold;'>Distributions des variables explicatives (X)</h6>",
                    unsafe_allow_html=True)

                # Affichage des histogrammes en 3 colonnes
                columns_per_row = 3
                for i, col in enumerate(st.session_state['X'].columns):
                    if i % columns_per_row == 0:
                        cols = st.columns(columns_per_row)

                    fig, ax = plt.subplots()
                    sns.histplot(st.session_state['X'][col], kde=True, ax=ax)
                    ax.set_title(f"Distribution de {col}")
                    cols[i % columns_per_row].pyplot(fig)

                st.markdown("---")  # Ligne de séparation

                # Visualisation des boxplots des variables explicatives (X) en fonction de la cible (y)
                st.markdown(
                    "<h6 style='color: #000000; font-weight: bold;'>Boxplots des variables explicatives (X) en fonction de la cible (y)</h6>",
                    unsafe_allow_html=True)

                # Affichage des boxplots en 3 colonnes
                for i, col in enumerate(st.session_state['X'].columns):
                    if i % columns_per_row == 0:
                        cols = st.columns(columns_per_row)

                    fig, ax = plt.subplots()
                    sns.boxplot(x=st.session_state['y'], y=col, data=selected_data, ax=ax)
                    ax.set_title(f"Boxplot de {col} en fonction de {st.session_state['y'].name}")
                    cols[i % columns_per_row].pyplot(fig)

        # Sous-onglet 2 : Matrice de corrélation & Pairplot
        with subtab2:
            # Titre de l'onglet
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Analyse des relations entre variables</h5>",
                        unsafe_allow_html=True)

            # Caption explicative pour cet onglet
            st.caption(
                "Dans cet onglet, nous analysons les relations entre les variables explicatives (X) à travers une matrice de corrélation et un pairplot. Cela permet d'identifier les relations linéaires et d'explorer visuellement les patterns entre variables.")

            # Titre avant la heatmap de corrélation
            st.markdown("<h6 style='color: #000000; font-weight: bold;'>Matrice de corrélation</h6>",
                        unsafe_allow_html=True)

            # Caption expliquant la matrice de corrélation
            st.caption(
                "La matrice de corrélation montre les coefficients de corrélation entre chaque paire de variables explicatives. Un coefficient proche de 1 indique une forte corrélation positive, tandis qu'un coefficient proche de -1 indique une forte corrélation négative.")

            # Vérifier que 'X' est bien initialisé et n'est pas None avant d'appeler la méthode corr()
            if st.session_state.get('X') is not None:
                # Assurez-vous que st.session_state['X'] est un DataFrame
                if not st.session_state['X'].empty:
                    corr_matrix = st.session_state['X'].corr()
                    # Afficher les données brutes dans un expander
                    with st.expander("Voir les données brutes de la matrice de corrélation"):
                        st.dataframe(corr_matrix)

                    # Afficher la heatmap de corrélation avec un titre
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title("Matrice de corrélation des variables explicatives (X)")
                    st.pyplot(fig)
                else:
                    st.warning("Le DataFrame X est vide, veuillez vérifier votre sélection des variables explicatives.")
            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives (X) dans l'onglet de sélection.")

            st.markdown("---")  # Ligne de séparation

            # Titre avant le pairplot
            st.markdown("<h6 style='color: #000000; font-weight: bold;'>Pairplot des variables explicatives (X)</h6>",
                        unsafe_allow_html=True)

            # Caption expliquant le pairplot
            st.caption(
                "Le pairplot visualise la relation entre chaque paire de variables explicatives. Il permet d'identifier des relations visuelles entre les variables, ainsi que des patterns potentiels comme la linéarité ou la dispersion.")

            # Vérifier que 'X' est bien un DataFrame valide avant d'appeler sns.pairplot()
            if st.session_state.get('X') is not None:
                # Assurez-vous que st.session_state['X'] est un DataFrame et non vide
                if isinstance(st.session_state['X'], pd.DataFrame) and not st.session_state['X'].empty:
                    # Afficher le pairplot
                    pairplot_fig = sns.pairplot(st.session_state['X'])
                    st.pyplot(pairplot_fig)
                else:
                    st.warning(
                        "Le DataFrame X est vide ou incorrect, veuillez vérifier votre sélection des variables explicatives.")
            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives (X) dans l'onglet de sélection.")

    # Sous-onglet 3 : Information sur les variables
    with subtab3:
        st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Description des variables ℹ (X et y)</h5>",
                    unsafe_allow_html=True)

        # Caption explicative pour ce sous-onglet
        st.caption(
            "Dans ce sous-onglet, nous affichons des informations détaillées sur chaque variable explicative (X) et la variable cible (y), qu'elle soit catégorielle ou numérique.")

        # Vérification si les variables explicatives (X) et la cible (y) sont disponibles
        if st.session_state.get('X') is not None and st.session_state.get('y') is not None:

            # Section pour les variables explicatives (X)
            st.markdown("<h6 style='color: #000000; font-weight: bold;'>Variables explicatives (X)</h6>",
                        unsafe_allow_html=True)

            # Tableau avec les statistiques importantes des variables explicatives (X)
            summary_data = []
            for col in st.session_state['X'].columns:
                summary_data.append({
                    'Variable': col,
                    'Type': str(df[col].dtype),
                    'Valeurs uniques': df[col].nunique(),
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Moyenne': df[col].mean(),
                    'Écart-type': df[col].std()
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)

            # Séparateur avant la variable cible
            st.markdown("---")

            # Section pour la variable cible (y)
            st.markdown("<h6 style='color: #000000; font-weight: bold;'>Variable cible (y)</h6>",
                        unsafe_allow_html=True)
            target_col = st.session_state['y'].name

            # Vérifier si la variable cible est catégorielle ou numérique
            if pd.api.types.is_numeric_dtype(st.session_state['y']):
                # Si y est numérique, afficher le tableau des statistiques numériques
                y_summary = {
                    'Variable': target_col,
                    'Type': str(df[target_col].dtype),
                    'Valeurs uniques': df[target_col].nunique(),
                }
                st.dataframe(pd.DataFrame([y_summary]))

            else:
                # Si y est catégorielle, afficher deux colonnes avec les valeurs textuelles et les fréquences
                col1, col2 = st.columns(2)

                with col1:
                    # Liste des catégories
                    st.markdown(f"**Liste des catégories de {target_col}** :")
                    st.write(df[target_col].unique().tolist())

                    # Tableau des fréquences des catégories
                    st.markdown(f"**Fréquence des catégories de {target_col}** :")
                    st.dataframe(df[target_col].value_counts())

                with col2:
                    # Diagramme circulaire pour la répartition en pourcentage des catégories
                    fig, ax = plt.subplots(figsize=(6, 6))  # Taille ajustée pour une meilleure visualisation

                    # Calcul des pourcentages
                    category_percentages = df[target_col].value_counts(normalize=True) * 100

                    # Création du pie chart
                    wedges, texts, autotexts = ax.pie(category_percentages,
                                                      labels=category_percentages.index,
                                                      autopct='%1.1f%%',
                                                      colors=sns.color_palette('pastel'),
                                                      startangle=90,
                                                      wedgeprops={'edgecolor': 'black'})

                    # Style des textes et pourcentages
                    for text in texts:
                        text.set_fontsize(10)
                    for autotext in autotexts:
                        autotext.set_fontsize(10)

                    ax.set_title(f"Répartition en pourcentage de {target_col} (Pie Chart)", fontsize=10,
                                 fontweight='bold')
                    st.pyplot(fig)

        else:
            st.warning("Aucune sélection de variables explicatives (X) et de cible (y) n'a été effectuée.")

    # Onglet 3 : Modélisation
    with tab3:

        # Création des sous-onglets
        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs(
            ["LazyPredict", "Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Autres modèles"]
        )

        # Sous-onglet 1 : LazyPredict
        with subtab1:
            st.caption(
                "LazyPredict permet une modélisation rapide en testant plusieurs modèles de classification sans nécessiter d'importante configuration préalable.")

            # Vérification que les variables X et y sont bien définies avant de continuer
            if st.session_state['X'] is not None and st.session_state['y'] is not None:
                X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'],
                                                                    test_size=test_size, random_state=random_state)

                if st.button("Lancer LazyPredict"):
                    # Lancer LazyPredict
                    lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True)
                    models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)

                    # Afficher un message pour l'utilisateur
                    st.success("LazyPredict s'est exécuté correctement. Voici les résultats des modèles.")

                    # Afficher les résultats sous forme de DataFrame
                    st.dataframe(models)

                    st.caption("""
                    - **Accuracy** : Le pourcentage de prédictions correctes parmi l'ensemble des prédictions. Plus il est élevé, meilleur est le modèle.
                    - **Balanced Accuracy** : Moyenne entre le taux de vrais positifs et de vrais négatifs. Utile pour les ensembles de données déséquilibrés.
                    - **ROC AUC** : Zone sous la courbe ROC (Receiver Operating Characteristic), qui mesure la capacité du modèle à discriminer entre classes. Ici, elle n'est pas disponible.
                    - **F1 Score** : Moyenne harmonique entre précision (precision) et rappel (recall). Un score F1 élevé indique un bon équilibre entre ces deux métriques.
                    - **Time Taken** : Temps nécessaire pour entraîner le modèle.
                    """)

                    # Extraire les 5 meilleurs modèles basés sur l'exactitude ('Accuracy')
                    top_5_models = models.sort_values(by="Accuracy", ascending=False).head(5)

                    # Extraire les 5 meilleurs modèles basés sur Accuracy, Balanced Accuracy, et F1 Score
                    top_5_accuracy = models.sort_values(by="Accuracy", ascending=False).head(5)
                    top_5_balanced_accuracy = models.sort_values(by="Balanced Accuracy", ascending=False).head(5)
                    top_5_f1_score = models.sort_values(by="F1 Score", ascending=False).head(5)

                    st.markdown("---")

                    # Afficher les résultats sur la même ligne dans un tableau
                    st.markdown(
                        "<h6 style='color: #000000; font-weight: bold;'>Top 5 des meilleurs modèles sur les différents KPI</h6>",
                        unsafe_allow_html=True)

                    # Construire un DataFrame pour afficher les meilleurs modèles sur les différentes colonnes
                    top_5_df = pd.DataFrame({
                        "Model (Accuracy)": top_5_accuracy.index,
                        "Accuracy": top_5_accuracy["Accuracy"].values,
                        "Model (Balanced Accuracy)": top_5_balanced_accuracy.index,
                        "Balanced Accuracy": top_5_balanced_accuracy["Balanced Accuracy"].values,
                        "Model (F1 Score)": top_5_f1_score.index,
                        "F1 Score": top_5_f1_score["F1 Score"].values
                    })

                    # Afficher le DataFrame dans l'application Streamlit
                    st.dataframe(top_5_df)

                    # Afficher un message expliquant les 5 meilleurs modèles
                    st.info(
                        "Les 5 meilleurs modèles affichés ici sont ceux ayant obtenu les scores les plus élevés pour le test LazyPredict.")

                    st.markdown("---")

                    # Affichage des meilleurs modèles par KPI sous 3 colonnes
                    st.markdown(
                        "<h6 style='color: #000000; font-weight: bold;'>Visualisation des 5 meilleurs modèles sur les différents KPI</h6>",
                        unsafe_allow_html=True)

                    # Affichage des meilleures visualisations dans des expanders
                    with st.expander("Voir la visualisation des 5 meilleurs modèles par Accurancy"):
                        # Visualisation pour Accuracy
                        st.markdown("**Top 5 Accuracy**")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.barplot(x=top_5_accuracy["Accuracy"], y=top_5_accuracy.index, palette="Blues_d", ax=ax)
                        ax.set_title("Accuracy", fontsize=8, fontweight="bold")
                        ax.set_ylabel('')  # Suppression de la légende sur l'axe des y
                        st.pyplot(fig)

                    with st.expander("Voir la visualisation des 5 meilleurs modèles par Balanced Accurancy"):
                        # Visualisation pour Balanced Accuracy
                        st.markdown("**Top 5 Balanced Accuracy**")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.barplot(x=top_5_balanced_accuracy["Balanced Accuracy"], y=top_5_balanced_accuracy.index,
                                    palette="Greens_d", ax=ax)
                        ax.set_title("Balanced Accuracy", fontsize=8, fontweight="bold")
                        ax.set_ylabel('')  # Suppression de la légende sur l'axe des y
                        st.pyplot(fig)

                    with st.expander("Voir la visualisation des 5 meilleurs modèles par F1 Score"):
                        # Visualisation pour F1 Score
                        st.markdown("**Top 5 F1 Score**")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.barplot(x=top_5_f1_score["F1 Score"], y=top_5_f1_score.index, palette="Oranges_d", ax=ax)
                        ax.set_title("F1 Score", fontsize=8, fontweight="bold")
                        ax.set_ylabel('')  # Suppression de la légende sur l'axe des y
                        st.pyplot(fig)


                else:
                    st.warning("Cliquez sur le bouton pour lancer LazyPredict.")
            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")

        # Sous-onglet 2 : Logistic Regression
        with subtab2:
            st.caption(
                "La régression logistique est un modèle statistique utilisé pour prédire la probabilité qu'un événement se produise.")

            # Vérification que les variables X et y sont bien définies avant de continuer
            if st.session_state['X'] is not None and st.session_state['y'] is not None:
                # Ajout des hyperparamètres ajustables par l'utilisateur
                st.markdown("<h6 style='color: #000000; font-weight: bold;'>Vous pouvez choisir vos hyperparamètres ici :</h6>",
                            unsafe_allow_html=True)

                # Hyperparamètres à ajuster
                max_iter = st.number_input("Max Iterations (nombre maximum d'itérations)", value=100, step=10)
                penalty = st.selectbox("Pénalité", options=['l2', 'none'], index=0,
                                       help="Choisissez la pénalité à utiliser.")
                C = st.slider("Paramètre de régularisation (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
                solver = st.selectbox("Algorithme d'optimisation (solver)", options=['lbfgs', 'saga', 'liblinear'],
                                      index=0, help="Choisissez l'algorithme d'optimisation.")

                # Instanciation du modèle avec les hyperparamètres
                model_lr = LogisticRegression(max_iter=max_iter, penalty=penalty, C=C, solver=solver)

                if st.button("Lancer Logistic Regression"):
                    # Split des données
                    X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'],
                                                                        test_size=test_size, random_state=random_state)
                    # Entraînement du modèle
                    model_lr.fit(X_train, y_train)

                    # Afficher un message de succès
                    st.success("Le modèle Logistic Regression a été entraîné avec succès.")

                    # Calcul des métriques
                    accuracy_lr = model_lr.score(X_test, y_test)
                    y_pred = model_lr.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True)

                    st.markdown("---")

                    # Affichage des métriques de performance sur 2 colonnes
                    col1, col2 = st.columns(2)

                    with col1:
                        # Encadrer le résultat de l'accuracy
                        st.markdown(f"""
                            <div style="padding: 10px; border: 2px solid #007BFF; border-radius: 5px; background-color: #F0F8FF">
                                <h6 style='color: #000000;'>Le score d'Accuracy obtenu pour ce modèle est de {accuracy_lr:.4f}</h6>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        # Inclure le rapport de classification dans un expander
                        with st.expander("Voir le rapport de classification"):
                            st.markdown("**Rapport de classification**")
                            st.dataframe(pd.DataFrame(report).transpose())

                    st.markdown("---")

                    # Visualisation de la matrice de confusion
                    st.markdown("**Matrice de confusion**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Vérités')
                    st.pyplot(fig)

                    # Caption explicatif pour la matrice de confusion
                    st.caption("""
                    La matrice de confusion est utilisée pour évaluer les performances d'un modèle de classification. 
                    Elle affiche le nombre de vraies prédictions et d'erreurs, réparties en 4 catégories :
                    - **Vrais Positifs (TP)** : Prédictions correctes pour la classe positive.
                    - **Faux Négatifs (FN)** : Étiquettes positives mal classées comme négatives.
                    - **Faux Positifs (FP)** : Étiquettes négatives mal classées comme positives.
                    - **Vrais Négatifs (TN)** : Prédictions correctes pour la classe négative.
                    L'objectif est de maximiser les vraies prédictions (TP et TN) tout en minimisant les erreurs (FP et FN).
                    """)

                else:
                    st.warning("Cliquez sur le bouton pour entraîner le modèle Logistic Regression.")
            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")

        # Sous-onglet 3 : Random Forest
        with subtab3:
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Random Forest</h5>", unsafe_allow_html=True)
            st.caption(
                "Le modèle Random Forest est un ensemble d'arbres de décision utilisés pour améliorer la précision et éviter le surapprentissage.")

            # Vérification que les variables X et y sont bien définies avant de continuer
            if st.session_state['X'] is not None and st.session_state['y'] is not None:
                # Ajout des hyperparamètres ajustables par l'utilisateur avec infobulles
                st.markdown(
                    "<h6 style='color: #000000; font-weight: bold;'>Vous pouvez choisir vos hyperparamètres ici :</h6>",
                    unsafe_allow_html=True)

                # Hyperparamètres à ajuster avec des infobulles pour chaque paramètre
                n_estimators = st.number_input(
                    "Nombre d'arbres (n_estimators)", value=100, step=10,
                    help="Le nombre total d'arbres dans la forêt. Plus le nombre est élevé, plus le modèle peut être précis, mais cela augmente également le temps de calcul.")

                max_depth = st.slider(
                    "Profondeur maximale des arbres (max_depth)", min_value=1, max_value=50, value=10, step=1,
                    help="La profondeur maximale des arbres. Limiter la profondeur des arbres permet d'éviter le surapprentissage (overfitting).")

                bootstrap = st.selectbox(
                    "Utiliser le bootstrap ?", options=[True, False], index=0,
                    help="Si activé, le bootstrap permet de sélectionner des échantillons avec remplacement pour entraîner les arbres. Cela améliore la robustesse du modèle.")

                # Instanciation du modèle avec les hyperparamètres
                model_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                                  random_state=random_state)

                if st.button("Lancer Random Forest"):
                    # Utilisation du spinner pour afficher un message de chargement pendant l'entraînement du modèle
                    with st.spinner('Entraînement du modèle Random Forest en cours...'):
                        # Split des données
                        X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'],
                                                                            test_size=test_size,
                                                                            random_state=random_state)
                        # Entraînement du modèle
                        model_rf.fit(X_train, y_train)

                    # Afficher un message de succès
                    st.success("Le modèle Random Forest a été entraîné avec succès.")

                    # Calcul des métriques
                    accuracy_rf = model_rf.score(X_test, y_test)
                    y_pred = model_rf.predict(X_test)
                    report_rf = classification_report(y_test, y_pred, output_dict=True)

                    st.markdown("---")

                    # Affichage des métriques de performance sur 2 colonnes
                    col1, col2 = st.columns(2)

                    with col1:
                        # Encadrer le résultat de l'accuracy
                        st.markdown(f"""
                            <div style="padding: 10px; border: 2px solid #007BFF; border-radius: 5px; background-color: #F0F8FF">
                                <h6 style='color: #000000;'>Le score d'Accuracy obtenu pour ce modèle est de {accuracy_rf:.4f}</h6>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        # Inclure le rapport de classification dans un expander
                        with st.expander("Voir le rapport de classification"):
                            st.markdown("**Rapport de classification**")
                            st.dataframe(pd.DataFrame(report_rf).transpose())

                    st.markdown("---")

                    # Visualisation de la matrice de confusion
                    st.markdown("**Matrice de confusion**")
                    cm_rf = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Vérités')
                    st.pyplot(fig)

                    # Caption explicatif pour la matrice de confusion
                    st.caption("""
                    La matrice de confusion est utilisée pour évaluer les performances d'un modèle de classification. 
                    Elle affiche le nombre de vraies prédictions et d'erreurs, réparties en 4 catégories :
                    - **Vrais Positifs (TP)** : Prédictions correctes pour la classe positive.
                    - **Faux Négatifs (FN)** : Étiquettes positives mal classées comme négatives.
                    - **Faux Positifs (FP)** : Étiquettes négatives mal classées comme positives.
                    - **Vrais Négatifs (TN)** : Prédictions correctes pour la classe négative.
                    L'objectif est de maximiser les vraies prédictions (TP et TN) tout en minimisant les erreurs (FP et FN).
                    """)

                else:
                    st.warning("Cliquez sur le bouton pour entraîner le modèle Random Forest.")
            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")

        # Sous-onglet 4 : K-Nearest Neighbors (K-NN)
        with subtab4:
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>K-Nearest Neighbors (K-NN)</h5>",
                        unsafe_allow_html=True)
            st.caption(
                "Le modèle K-Nearest Neighbors (K-NN) est un algorithme simple basé sur la distance pour classer ou prédire en fonction des points voisins les plus proches.")

            # Vérification que les variables X et y sont bien définies avant de continuer
            if st.session_state['X'] is not None and st.session_state['y'] is not None:
                # Ajout des hyperparamètres ajustables par l'utilisateur avec infobulles
                st.markdown(
                    "<h6 style='color: #000000; font-weight: bold;'>Vous pouvez choisir vos hyperparamètres ici :</h6>",
                    unsafe_allow_html=True)

                # Hyperparamètres à ajuster avec des infobulles pour chaque paramètre
                n_neighbors = st.number_input(
                    "Nombre de voisins (n_neighbors)", value=5, min_value=1, step=1,
                    help="Le nombre de voisins à prendre en compte pour la classification. Plus le nombre est élevé, plus les prédictions peuvent être stables, mais cela peut aussi rendre le modèle moins précis.")

                weights = st.selectbox(
                    "Type de pondération (weights)", options=['uniform', 'distance'], index=0,
                    help="Choisissez entre une pondération uniforme ou basée sur la distance pour les voisins les plus proches.")

                algorithm = st.selectbox(
                    "Algorithme de recherche (algorithm)", options=['auto', 'ball_tree', 'kd_tree', 'brute'], index=0,
                    help="L'algorithme utilisé pour calculer les voisins les plus proches. Par défaut, 'auto' choisit l'algorithme le plus approprié en fonction des données.")

                leaf_size = st.number_input(
                    "Taille des feuilles (leaf_size)", value=30, min_value=1, step=1,
                    help="La taille des feuilles à utiliser dans les algorithmes 'ball_tree' et 'kd_tree'. Elle influe sur la vitesse de construction et de requête.")

                # Instanciation du modèle avec les hyperparamètres
                model_knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                                 leaf_size=leaf_size)

                if st.button("Lancer K-NN"):
                    # Split des données
                    X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'],
                                                                        test_size=test_size, random_state=random_state)
                    # Entraînement du modèle
                    model_knn.fit(X_train, y_train)

                    # Afficher un message de succès
                    st.success("Le modèle K-NN a été entraîné avec succès.")

                    # Calcul des métriques
                    accuracy_knn = model_knn.score(X_test, y_test)
                    y_pred = model_knn.predict(X_test)
                    report_knn = classification_report(y_test, y_pred, output_dict=True)

                    st.markdown("---")

                    # Affichage des métriques de performance sur 2 colonnes
                    col1, col2 = st.columns(2)

                    with col1:
                        # Encadrer le résultat de l'accuracy
                        st.markdown(f"""
                            <div style="padding: 10px; border: 2px solid #007BFF; border-radius: 5px; background-color: #F0F8FF">
                                <h6 style='color: #000000;'>Le score d'Accuracy obtenu pour ce modèle est de {accuracy_knn:.4f}</h6>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        # Inclure le rapport de classification dans un expander
                        with st.expander("Voir le rapport de classification"):
                            st.markdown("**Rapport de classification**")
                            st.dataframe(pd.DataFrame(report_knn).transpose())

                    st.markdown("---")

                    # Visualisation de la matrice de confusion
                    st.markdown("**Matrice de confusion**")
                    cm_knn = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Vérités')
                    st.pyplot(fig)

                    # Caption explicatif pour la matrice de confusion
                    st.caption("""
                    La matrice de confusion est utilisée pour évaluer les performances d'un modèle de classification. 
                    Elle affiche le nombre de vraies prédictions et d'erreurs, réparties en 4 catégories :
                    - **Vrais Positifs (TP)** : Prédictions correctes pour la classe positive.
                    - **Faux Négatifs (FN)** : Étiquettes positives mal classées comme négatives.
                    - **Faux Positifs (FP)** : Étiquettes négatives mal classées comme positives.
                    - **Vrais Négatifs (TN)** : Prédictions correctes pour la classe négative.
                    L'objectif est de maximiser les vraies prédictions (TP et TN) tout en minimisant les erreurs (FP et FN).
                    """)

                else:
                    st.warning("Cliquez sur le bouton pour entraîner le modèle K-NN.")
            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")

        # Sous-onglet 5 : Autres modèles
        with subtab5:
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Autres modèles</h5>", unsafe_allow_html=True)
            st.caption(
                "Cet espace vous permet d'explorer d'autres modèles de classification ou de tester des modèles personnalisés.")

            # Liste des modèles disponibles
            model_options = ["SVM", "Gradient Boosting", "AdaBoost", "Decision Tree", "Naive Bayes"]

            # Sélection du modèle à tester
            selected_model = st.selectbox("Choisissez un modèle à tester", options=model_options)

            # Vérification que les variables X et y sont bien définies avant de continuer
            if st.session_state['X'] is not None and st.session_state['y'] is not None:

                # En fonction du modèle sélectionné, afficher les hyperparamètres spécifiques
                if selected_model == "SVM":
                    st.markdown("<h6 style='color: #000000; font-weight: bold;'>SVM - Support Vector Machine</h6>",
                                unsafe_allow_html=True)
                    C = st.slider("Paramètre de régularisation (C)", min_value=0.01, max_value=10.0, value=1.0,
                                  step=0.1,
                                  help="Contrôle la régularisation. Un petit C donne lieu à un modèle régularisé plus fort.")
                    kernel = st.selectbox("Kernel", options=["linear", "poly", "rbf", "sigmoid"],
                                          help="Le type de fonction de noyau à utiliser pour la séparation des classes.")
                    gamma = st.slider("Gamma", min_value=0.001, max_value=1.0, value=0.1, step=0.001,
                                      help="Définit l'influence d'un point d'entraînement sur la décision finale.")

                    model = SVC(C=C, kernel=kernel, gamma=gamma)

                elif selected_model == "Gradient Boosting":
                    st.markdown("<h6 style='color: #000000; font-weight: bold;'>Gradient Boosting</h6>",
                                unsafe_allow_html=True)
                    n_estimators = st.number_input("Nombre d'arbres (n_estimators)", value=100, step=10)
                    learning_rate = st.slider("Taux d'apprentissage (learning_rate)", min_value=0.01, max_value=1.0,
                                              value=0.1,
                                              step=0.01, help="Détermine l'impact de chaque arbre sur le modèle final.")
                    max_depth = st.slider("Profondeur maximale (max_depth)", min_value=1, max_value=10, value=3,
                                          help="La profondeur maximale des arbres permet de contrôler la complexité du modèle.")

                    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                       max_depth=max_depth)

                elif selected_model == "AdaBoost":
                    st.markdown("<h6 style='color: #000000; font-weight: bold;'>AdaBoost</h6>",
                                unsafe_allow_html=True)
                    n_estimators = st.number_input("Nombre d'arbres (n_estimators)", value=50, step=10,
                                                   help="Le nombre total de classificateurs faibles à combiner.")
                    learning_rate = st.slider("Taux d'apprentissage (learning_rate)", min_value=0.01, max_value=1.0,
                                              value=1.0,
                                              step=0.01,
                                              help="Contrôle la contribution de chaque classificateur faible.")
                    algorithm = st.selectbox("Algorithme", options=["SAMME", "SAMME.R"], index=1,
                                             help="Le type d'algorithme utilisé pour combiner les classificateurs.")

                    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                               algorithm=algorithm)

                elif selected_model == "Decision Tree":
                    st.markdown("<h6 style='color: #000000; font-weight: bold;'>Decision Tree</h6>",
                                unsafe_allow_html=True)
                    criterion = st.selectbox("Critère", options=["gini", "entropy"],
                                             help="La fonction à mesurer la qualité de la séparation.")
                    max_depth = st.slider("Profondeur maximale (max_depth)", min_value=1, max_value=20, value=5,
                                          help="La profondeur maximale de l'arbre contrôle la taille des sous-arbres.")
                    min_samples_split = st.slider("Min. échantillons pour diviser (min_samples_split)", min_value=2,
                                                  max_value=10,
                                                  value=2, step=1,
                                                  help="Le nombre minimum d'échantillons nécessaires pour diviser un nœud.")

                    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                   min_samples_split=min_samples_split)

                elif selected_model == "Naive Bayes":
                    st.markdown("<h6 style='color: #000000; font-weight: bold;'>Naive Bayes</h6>",
                                unsafe_allow_html=True)
                    var_smoothing = st.slider("Lissage des variances (var_smoothing)", min_value=1e-9, max_value=1e-5,
                                              value=1e-9,
                                              step=1e-9, format="%.9f",
                                              help="Ajuste la stabilité numérique en ajoutant une petite constante à la variance calculée.")
                    model = GaussianNB(var_smoothing=var_smoothing)

                # Entraîner le modèle si le bouton est cliqué
                if st.button(f"Lancer {selected_model}"):
                    # Split des données
                    X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'],
                                                                        test_size=test_size, random_state=random_state)
                    # Entraînement du modèle
                    model.fit(X_train, y_train)

                    # Afficher un message de succès
                    st.success(f"Le modèle {selected_model} a été entraîné avec succès.")

                    # Calcul des métriques
                    accuracy = model.score(X_test, y_test)
                    y_pred = model.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True)

                    st.markdown("---")

                    # Affichage des métriques de performance sur 2 colonnes
                    col1, col2 = st.columns(2)

                    with col1:
                        # Encadrer le résultat de l'accuracy
                        st.markdown(f"""
                            <div style="padding: 10px; border: 2px solid #007BFF; border-radius: 5px; background-color: #F0F8FF">
                                <h6 style='color: #000000;'>Le score d'Accuracy obtenu pour ce modèle est de {accuracy:.4f}</h6>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        # Inclure le rapport de classification dans un expander
                        with st.expander("Voir le rapport de classification"):
                            st.markdown("**Rapport de classification**")
                            st.dataframe(pd.DataFrame(report).transpose())

                    st.markdown("---")

                    # Visualisation de la matrice de confusion
                    st.markdown("**Matrice de confusion**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Vérités')
                    st.pyplot(fig)

                    # Caption explicatif pour la matrice de confusion
                    st.caption("""
                    La matrice de confusion est utilisée pour évaluer les performances d'un modèle de classification. 
                    Elle affiche le nombre de vraies prédictions et d'erreurs, réparties en 4 catégories :
                    - **Vrais Positifs (TP)** : Prédictions correctes pour la classe positive.
                    - **Faux Négatifs (FN)** : Étiquettes positives mal classées comme négatives.
                    - **Faux Positifs (FP)** : Étiquettes négatives mal classées comme positives.
                    - **Vrais Négatifs (TN)** : Prédictions correctes pour la classe négative.
                    """)

            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")

    # Onglet 4 : Évaluation
    with tab4:
        st.subheader("Évaluation des modèles")
        # Evaluation des modèles basés sur les résultats de l'entraînement
        st.write("À implémenter : Sélectionner un modèle pour l'évaluation.")

    # Onglet 5 : Comparaison (vide pour l'instant)
    with tab5:
        st.subheader("Comparaison des modèles")
        st.write("Fonctionnalité à venir.")
