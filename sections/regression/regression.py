import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lazypredict.Supervised import LazyRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

def regression_page():
    st.caption("Bienvenue dans le Playground de Regression")

    # Jauge de test_size et random_state dans la barre latérale
    test_size = st.sidebar.slider("Proportion du Test (en %)", min_value=5, max_value=50, value=20, step=1) / 100
    random_state = st.sidebar.number_input("Random State", value=42)

    # Ajouter un bouton dans la sidebar pour lancer LazyRegressor
    st.sidebar.caption("Le LazyRegressor permet de déterminer rapidement les meilleurs modèles pour évaluer vos données.")
    if st.sidebar.button("Lancer un LazyRegressor"):
        st.session_state.selected_tab = "LazyRegressor"
        st.session_state.run_lazy_regressor = True
    else:
        st.session_state.run_lazy_regressor = False

    # Création des onglets principaux
    apercu_tab, modeles_tab, visuel_tab = st.tabs(["Aperçu des Données", "Modèles", "Visuels"])

    with apercu_tab:
        st.subheader("Aperçu des Données")

        # Chargement des données depuis le local
        df = pd.read_csv("data/diabete.csv", index_col=0)
        st.write(df)

        # Préparation des données
        X = df.drop(columns=['target'])
        y = df['target']

        # Séparation des données d'entraînement et de test avec test_size et random_state depuis la sidebar
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Description des variables
        descriptions = {
            'Age': 'Age',
            'Sexe': 'Homme ou Femme',
            'BMI': 'Body Mass Index',
            'Bp': 'Blood pressure',
            's1': 'Variable explicative 1',
            's2': 'Variable explicative 2',
            's3': 'Variable explicative 3',
            's4': 'Variable explicative 4',
            's5': 'Variable explicative 5',
            's6': 'Variable explicative 6',
        }

        # Affichage des descriptions des colonnes
        st.subheader("Descriptions des Colonnes")
        for column, description in descriptions.items():
            st.write(f"**{column}:** {description}")

    with modeles_tab:
        # Création des sous-onglets pour chaque modèle
        lasso_tab, linear_regression_tab, extra_trees_tab = st.tabs(["Modèle Lasso", "Régression Linéaire", "Extra Trees Regressor"])

        with lasso_tab:
            st.subheader("Modèle Lasso")
            if st.button("Évaluer Modèle Lasso"):
                # Modèle Lasso avec GridSearchCV
                param_grid = {'alpha': np.logspace(-4, 4, 100)}
                lasso = Lasso()
                grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5,
                                           scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_alpha = grid_search.best_params_['alpha']
                best_lasso = grid_search.best_estimator_
                y_pred = best_lasso.predict(X_test)
                mse_lasso = mean_squared_error(y_test, y_pred)
                r2_lasso = r2_score(y_test, y_pred)

                st.subheader("Évaluation du Modèle Lasso avec un GridSearch")
                st.write(f"Meilleure Alpha: {best_alpha}")
                st.write(f"Mean Squared Error (MSE): {mse_lasso}")
                st.write(f"R^2 Score: {r2_lasso}")

                comparaison = pd.DataFrame({
                    'Valeur Réelle (y_test)': y_test,
                    'Valeur Prédite (y_pred)': y_pred
                })
                st.subheader("Comparaison entre la prédiction et la target")
                st.dataframe(comparaison)

                # Utiliser des colonnes pour afficher les graphiques côte à côte
                col1, col2 = st.columns(2)

                with col1:

                    residuals = y_test - y_pred
                    fig, ax = plt.subplots()
                    ax.scatter(y_pred, residuals, alpha=0.5)
                    ax.axhline(y=0, color='red', linestyle='--')
                    ax.set_xlabel('Valeurs Prédites (y_pred)')
                    ax.set_ylabel('Résidus')
                    ax.set_title('Graphique des Résidus pour Régression Lasso', loc='center')
                    st.pyplot(fig)

                with col2:

                    alphas, coefs, _ = lasso_path(X_train, y_train, alphas=np.logspace(-4, 4, 100))
                    fig, ax = plt.subplots()
                    for i in range(coefs.shape[0]):
                        ax.plot(np.log10(alphas), coefs[i, :], label=X.columns[i])
                    ax.set_xlabel('Log10(Alpha)')
                    ax.set_ylabel('Coefficients')
                    ax.set_title('Chemin de Régularisation (Lasso Path)', loc='center')
                    ax.legend()
                    st.pyplot(fig)

        with linear_regression_tab:
            st.subheader("Régression Linéaire")
            if st.button("Évaluer Régression Linéaire"):
                # Modèle de régression linéaire
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                comparaison = pd.DataFrame({
                    'Valeur Réelle (y_test)': y_test,
                    'Valeur Prédite (y_pred)': y_pred
                })
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.subheader("Évaluation du Modèle de Régression Linéaire")
                st.write(f"Mean Squared Error (MSE): {mse}")
                st.write(f"R^2 Score: {r2}")
                st.subheader("Comparaison entre la prédiction et la target")
                st.dataframe(comparaison)

                # Utiliser des colonnes pour afficher les graphiques côte à côte
                col1, col2 = st.columns(2)

                with col1:

                    fig, ax = plt.subplots()
                    ax.scatter(comparaison['Valeur Réelle (y_test)'], comparaison['Valeur Prédite (y_pred)'], alpha=0.5,
                               label='Prédictions')
                    ax.plot([min(comparaison['Valeur Réelle (y_test)']), max(comparaison['Valeur Réelle (y_test)'])],
                            [min(comparaison['Valeur Réelle (y_test)']), max(comparaison['Valeur Réelle (y_test)'])],
                            color='red', linestyle='--', label='Référence')
                    ax.set_xlabel('Valeurs Réelles (y_test)')
                    ax.set_ylabel('Valeurs Prédites (y_pred)')
                    ax.set_title('Graphique de Dispersion', loc='center')  # Centre le titre
                    ax.legend()
                    st.pyplot(fig)

                with col2:

                    fig, ax = plt.subplots()
                    sns.regplot(x='Valeur Réelle (y_test)', y='Valeur Prédite (y_pred)', data=comparaison,
                                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
                    ax.set_xlabel('Valeurs Réelles (y_test)')
                    ax.set_ylabel('Valeurs Prédites (y_pred)')
                    ax.set_title('Graphique de Régression', loc='center')  # Centre le titre
                    st.pyplot(fig)

        with extra_trees_tab:
            st.subheader("Extra Trees Regressor")
            # Paramètres pour ExtraTreesRegressor
            n_estimators = st.slider("Nombre d'arbres (n_estimators)", min_value=10, max_value=500, value=100, step=10)
            max_depth = st.slider("Profondeur maximale (max_depth)", min_value=1, max_value=40, value=10)
            bootstrap = st.selectbox("Utiliser Bootstrap ?", [False, True])
            if st.button("Évaluer Extra Trees Regressor"):
                # Utiliser les paramètres définis
                model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, random_state=random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                comparaison = pd.DataFrame({
                    'Valeur Réelle (y_test)': y_test,
                    'Valeur Prédite (y_pred)': y_pred
                })

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Affichage des résultats pour Extra Trees Regressor
                st.subheader("Évaluation du Modèle Extra Trees Regressor")
                st.write(f"Mean Squared Error (MSE): {mse}")
                st.write(f"R^2 Score: {r2}")

                st.subheader("Comparaison entre la prédiction et la target")
                st.dataframe(comparaison)

    with visuel_tab:
        # Création des sous-onglets dans l'onglet Visuels
        visuel_subtabs = st.tabs(["LazyRegressor", "Matrice de Corrélation"])

        with visuel_subtabs[0]:
            if st.session_state.run_lazy_regressor:
                # LazyRegressor pour déterminer les meilleurs modèles
                reg = LazyRegressor()
                models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                st.subheader("LazyRegressor pour déterminer les meilleurs modèles")
                st.dataframe(models)

        with visuel_subtabs[1]:
            # Matrice de corrélation
            corr_matrix = df.corr()
            st.subheader("Matrice de Corrélation")

            # Affichage de la heatmap
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)


