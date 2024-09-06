# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from lazypredict.Supervised import LazyClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix
#
# # Charger le fichier CSV depuis le local
# @st.cache_data
# def load_data():
#     return pd.read_csv("./data/vin.csv", index_col=0)
#
# df = load_data()
#
# st.title("Projet Machine Learning 🎈")
#
# # Séparer les variables explicatives (X) et la variable cible (y)
# X = df.drop(columns=['target'])
# y = df['target']
#
# # Page principale de classification
# def classification_page():
#     st.caption("Bienvenue dans le Playground de Classification")
#
#     # Sélection des paramètres dans la sidebar
#     st.sidebar.markdown("### Paramètres")
#
#     # Slider pour la taille du jeu de test
#     test_size = st.sidebar.slider(
#         "Taille du jeu de test (%)", 5, 50, 20, help="Entrez une valeur comprise entre 5 et 50"
#     ) / 100
#
#     # Nombre aléatoire pour reproduire les résultats
#     random_state = st.sidebar.number_input("Random state", value=42, help="A définir pour reproduire les résultats")
#
#     # Séparation des données d'entraînement et de test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#
#     # Affichage de la taille des ensembles d'entraînement et de test
#     st.subheader("Taille des ensembles d'entraînement et de test")
#     st.write(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
#     st.write(f"Taille de l'ensemble de test : {X_test.shape[0]}")
#
#     # Sélection des variables explicatives par l'utilisateur
#     st.sidebar.markdown("## Sélection des variables explicatives")
#     selected_columns = st.sidebar.multiselect("Sélectionnez les variables explicatives", options=X.columns.tolist(),
#                                               default=X.columns.tolist())
#
#     if selected_columns:
#         X_train = X_train[selected_columns]
#         X_test = X_test[selected_columns]
#     else:
#         st.warning("Aucune colonne sélectionnée pour l'entraînement.")
#
#     # Initialiser y_pred et report
#     y_pred = None
#     report = None
#
#     # Création des onglets
#     tab1, tab2, tab3, tab4, tab5 = st.tabs(
#         ["Aperçu du dataset", "LazyPredict", "Choix du modèle", "Prévisualisation du modèle", "Comparaison des modèles"]
#     )
#
#     # Onglet 1 : Aperçu du dataset
#     with tab1:
#         st.subheader("Aperçu du dataset avant la phase de Machine Learning")
#         st.write(df)
#
#     # Onglet 2 : LazyPredict
#     with tab2:
#         st.subheader("LazyPredict: Comparaison rapide de plusieurs modèles")
#         if st.button("Lancer LazyPredict"):
#             lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
#             models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
#             st.write("Voici les résultats des différents modèles :")
#             st.dataframe(models)
#
#     # Onglet 3 : Choix du modèle ML
#     with tab3:
#         st.subheader("Choisissez un modèle pour l'entraînement")
#
#         model_choice = st.selectbox("Choisissez un modèle",
#                                     ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"])
#
#         # Paramétrage selon le modèle choisi
#         if model_choice == "Logistic Regression":
#             max_iter = st.number_input("Max Iterations", value=100)
#             model = LogisticRegression(max_iter=max_iter)
#         elif model_choice == "Random Forest":
#             n_estimators = st.number_input("Nombre d'arbres", value=100)
#             model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
#         elif model_choice == "K-Nearest Neighbors":
#             n_neighbors = st.number_input("Nombre de voisins", value=5)
#             model = KNeighborsClassifier(n_neighbors=n_neighbors)
#
#         # Entraîner le modèle
#         if st.button(f"Lancer {model_choice}"):
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             report = classification_report(y_test, y_pred, output_dict=True)
#             st.write(f"Rapport de classification pour {model_choice} :")
#             st.json(report)
#
#     # Onglet 4 : Prévisualisation du modèle
#     with tab4:
#         st.subheader("Prévisualisation des résultats")
#
#         # Vérifier si y_pred a été généré
#         if y_pred is not None:
#             cm = confusion_matrix(y_test, y_pred)
#             st.write("Matrice de confusion :")
#             fig, ax = plt.subplots()
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#             st.pyplot(fig)
#         else:
#             st.write("Aucune prédiction n'a été faite pour afficher la matrice de confusion.")
#
#     # Onglet 5 : Comparaison des modèles
#     with tab5:
#         st.subheader("Comparaison des modèles")
#         st.write("À venir.")
#
# # Exécuter la page de classification
# classification_page()
#
# with tab1:
#     st.subheader("Prétraitement des données")
#
#     # Sélection des colonnes explicatives et de la cible
#     features = st.multiselect("Sélectionnez les features (X)", data.columns.tolist())
#     target = st.selectbox("Sélectionnez la cible (y)", data.columns.tolist())
#
#     if not features or not target:
#         st.warning("Veuillez sélectionner les colonnes pour X et y.")
#         return
#
#     X = data[features]
#     y = data[target]
#
#     # Split et seed dans la sidebar
#     test_size = st.sidebar.slider("Taille du jeu de test (%)", min_value=10, max_value=50, value=30) / 100
#     random_state = st.sidebar.slider("Seed pour la reproduction", min_value=0, max_value=100, value=42)
#
#     # Normalisation des données (facultatif)
#     if st.checkbox("Appliquer la normalisation des données"):
#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)
#
# with tab2:
#     st.subheader("Modélisation")
#
#     # LazyPredict pour les utilisateurs débutants
#     if st.checkbox("Utiliser LazyPredict pour voir les modèles de base"):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#         lazy_clf = LazyClassifier()
#         models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
#         st.write("### Comparaison des modèles :")
#         st.write(models)
#
#     st.write("### Choisissez un modèle de classification :")
#     model_choice = st.selectbox("Modèle", ["Logistic Regression", "K-NN", "Random Forest"])
#
#     # Choix des modèles et options
#     if model_choice == "Logistic Regression":
#         model = LogisticRegression()
#     elif model_choice == "K-NN":
#         n_neighbors = st.slider("Nombre de voisins (k)", min_value=1, max_value=20, value=5)
#         model = KNeighborsClassifier(n_neighbors=n_neighbors)
#     elif model_choice == "Random Forest":
#         n_estimators = st.slider("Nombre d'arbres", min_value=10, max_value=100, value=50)
#         model = RandomForestClassifier(n_estimators=n_estimators)
#
#     # Bouton pour exécuter le modèle
#     if st.button("Run le modèle"):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#         model.fit(X_train, y_train)
#         st.success("Le modèle a été entraîné avec succès.")
#
# with tab3:
#     st.subheader("Évaluation")
#
#     if st.button("Évaluer le modèle"):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#         y_pred = model.predict(X_test)
#
#         # Matrice de confusion
#         st.write("#### Matrice de confusion")
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         fig, ax = plt.subplots()
#         sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
#         st.pyplot(fig)
#
#         # Rapport de classification
#         st.write("#### Rapport de classification")
#         report = classification_report(y_test, y_pred, output_dict=True)
#         st.write(pd.DataFrame(report).transpose())
#
# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from lazypredict.Supervised import LazyClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Vérifier si df est déjà chargé dans st.session_state
# if 'df' not in st.session_state:
#     # Charger le fichier CSV dans df
#     st.session_state['df'] = pd.read_csv("./data/vin.csv").drop(columns=['Unnamed: 0'])
# df = st.session_state['df']  # Récupérer les données depuis st.session_state
#
# # Fonction principale pour la classification
# def classification_page():
#     st.caption("Playground ML - Classification")
#
#     # Création des onglets principaux
#     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
#         ["Sélection des variables", "Aperçu du dataset", "Modélisation LazyPredict",
#          "Modélisation Autres", "Évaluation", "Comparaison"]
#     )
#
#     # Onglet 1 : Sélection des variables
#     with tab1:
#         st.subheader("Sélection des variables explicatives (X) et de la cible (y)")
#
#         # Sélection des variables explicatives par l'utilisateur
#         selected_columns = st.multiselect("Sélectionnez les variables explicatives (X)", options=df.columns.tolist(), default=[])
#
#         # Sélection de la cible (y) sans sélection par défaut
#         target = st.selectbox("Sélectionnez la cible (y)", options=[col for col in df.columns if col not in selected_columns], index=0)
#
#         # Stocker X et y dans st.session_state
#         if selected_columns and target:
#             st.session_state['X'] = df[selected_columns]
#             st.session_state['y'] = df[target]
#
#             st.write("### Variables explicatives (X) sélectionnées :")
#             st.write(selected_columns)
#             st.write("### Cible (y) sélectionnée :")
#             st.write(target)
#
#             # Bouton pour valider le choix
#             if st.button("Valider votre choix"):
#                 confirm = st.radio("Voulez-vous valider votre choix ?", ("Oui", "Non"))
#
#                 if confirm == "Oui":
#                     st.success("Votre sélection a été validée.")
#                 elif confirm == "Non":
#                     st.warning("Veuillez refaire votre sélection.")
#
#         else:
#             st.warning("Veuillez sélectionner les colonnes pour X et y.")
#
#     # Onglet 2 : Aperçu du dataset
#     with tab2:
#         st.subheader("Aperçu du dataset")
#         #
#         # if st.session_state['X'] is not None and st.session_state['y'] is not None:
#         #     # Appliquer des couleurs pour différencier les colonnes X et la cible y
#         #     styled_data = df.style.apply(
#         #         lambda x: ['background-color: lightblue' if col in st.session_state['X'].columns else
#         #                    'background-color: lightgreen' if col == st.session_state['y'].name else ''
#         #                    for col in df.columns], axis=1
#         #     )
#         #     st.dataframe(styled_data)
#         # else:
#         #     st.write(df)  # Affiche les données sans coloration si aucune sélection n'est faite
#
#     # Onglet 3 : Modélisation LazyPredict
#     with tab3:
#         st.subheader("Modélisation rapide avec LazyPredict")
#
#         # # Vérifier que les variables X et y sont bien définies avant de continuer
#         # if st.session_state['X'] is not None and st.session_state['y'] is not None:
#         #     test_size = st.slider("Taille du jeu de test (%)", 5, 50, 20) / 100
#         #     random_state = st.number_input("Random state", value=42)
#         #
#         #     X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'],
#         #                                                         test_size=test_size, random_state=random_state)
#         #
#         #     if st.button("Lancer LazyPredict"):
#         #         lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True)
#         #         models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
#         #         st.write("Résultats LazyPredict :")
#         #         st.dataframe(models)
#         # else:
#         #     st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")
#
#     # Onglet 4 : Modélisation Autres avec sous-onglets
#     with tab4:
#         st.subheader("Modélisation Avancée")
#
#         # Vérifier que les variables X et y sont bien définies avant de continuer
#         if st.session_state['X'] is not None and st.session_state['y'] is not None:
#             subtab1, subtab2, subtab3, subtab4 = st.tabs(
#                 ["Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Autres"]
#             )
#
#             # Sous-onglet Logistic Regression
#             with subtab1:
#                 st.subheader("Logistic Regression")
#                 max_iter = st.number_input("Max Iterations", value=100)
#                 model_lr = LogisticRegression(max_iter=max_iter)
#                 if st.button("Entraîner Logistic Regression"):
#                     model_lr.fit(X_train, y_train)
#                     st.write("Modèle Logistic Regression entraîné.")
#
#             # Sous-onglet Random Forest
#             with subtab2:
#                 st.subheader("Random Forest")
#                 n_estimators = st.number_input("Nombre d'arbres", value=100)
#                 model_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
#                 if st.button("Entraîner Random Forest"):
#                     model_rf.fit(X_train, y_train)
#                     st.write("Modèle Random Forest entraîné.")
#
#             # Sous-onglet K-Nearest Neighbors
#             with subtab3:
#                 st.subheader("K-Nearest Neighbors")
#                 n_neighbors = st.number_input("Nombre de voisins", value=5)
#                 model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
#                 if st.button("Entraîner K-NN"):
#                     model_knn.fit(X_train, y_train)
#                     st.write("Modèle K-NN entraîné.")
#
#             # Sous-onglet Autres modèles
#             with subtab4:
#                 st.subheader("Autres modèles")
#                 st.write("Sélectionnez et entraînez d'autres modèles à tester.")
#         else:
#             st.warning("Veuillez d'abord sélectionner les variables explicatives et la cible dans l'onglet 1.")
#
#     # Onglet 5 : Évaluation
#     with tab5:
#         st.subheader("Évaluation des modèles")
#         # Evaluation des modèles basés sur les résultats de l'entraînement
#         st.write("À implémenter : Sélectionner un modèle pour l'évaluation.")
#
#     # Onglet 6 : Comparaison (vide pour l'instant)
#     with tab6:
#         st.subheader("Comparaison des modèles")
#         st.write("Fonctionnalité à venir.")
#
# # Exécuter la page de classification
# classification_page()