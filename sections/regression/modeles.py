import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from lazypredict.Supervised import LazyRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Fonction principale pour l'affichage des modeles
def show_modeles(X_train, X_test, y_train, y_test, random_state):
    # Séparation des onglets pour chaque modèle
    lazy_tab, lasso_tab, linear_regression_tab, extra_trees_tab, xgboost_tab = st.tabs(
        ["LazyRegressor", "Régression Lasso", "Régression Linéaire", "Extra Trees Regressor", "XGBoost"])

    # Appel de chaque modèle dans son onglet
    with lazy_tab:
        show_lazy_regressor(X_train, X_test, y_train, y_test)

    with lasso_tab:
        show_lasso_model(X_train, X_test, y_train, y_test)

    with linear_regression_tab:
        show_linear_regression(X_train, X_test, y_train, y_test)

    with extra_trees_tab:
        show_extra_trees(X_train, X_test, y_train, y_test, random_state)

    with xgboost_tab:
        show_xgboost(X_train, X_test, y_train, y_test, random_state)

# Fonction pour l'affichage du learning curve cbof
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error',
                                                            n_jobs=-1)
    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, label='Erreur d\'Entraînement', marker='o')
    ax.plot(train_sizes, test_mean, label='Erreur de Validation', marker='o')
    ax.set_xlabel('Taille de l\'échantillon d\'apprentissage')
    ax.set_ylabel('Erreur Quadratique Moyenne')
    ax.set_title('Courbe d\'Apprentissage')
    ax.legend()

    return fig

# Fonction pour l'affichage du LazyRegressor
def show_lasso_model(X_train, X_test, y_train, y_test):
    st.subheader("Régression Lasso")

    # Initialisation des variables de session
    if 'grid_search' not in st.session_state:
        st.session_state.grid_search = None
        st.session_state.best_alpha = None
        st.session_state.selected_alpha = None

    st.caption("Gridsearch permet de déterminer les meilleurs hyperparamètres d'alpha pour le modèle")

    # Bouton pour lancer le GrdSearch
    if st.button("Lancer un GridSearch"):
        with st.spinner("Recherche des hyperparamètres en cours..."):
            param_grid = {'alpha': np.logspace(-4, 4, 100)}
            lasso = Lasso()
            grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            st.session_state.grid_search = grid_search
            st.session_state.best_alpha = grid_search.best_params_['alpha']
            st.session_state.selected_alpha = st.session_state.best_alpha
        st.success(f"Recherche terminée! Meilleure Alpha: {st.session_state.best_alpha}")

    if st.session_state.grid_search is not None and st.session_state.best_alpha is not None:
        st.write(f"Meilleure Alpha trouvée: {st.session_state.best_alpha}")

        # Input pour modifier l'alpha au choix
        alpha_input = st.text_input(
            'Entrez une valeur d\'alpha',
            value=str(round(st.session_state.best_alpha, 4))
        )

        # Try except pas tres utile
        try:
            alpha = float(alpha_input)
            if alpha <= 0:
                st.error("L'alpha doit être un nombre positif.")
            else:
                if st.button("Évaluer Régression Lasso"):
                    lasso = Lasso(alpha=alpha)
                    lasso.fit(X_train, y_train)
                    y_pred = lasso.predict(X_test)
                    mse_lasso = mean_squared_error(y_test, y_pred)
                    r2_lasso = r2_score(y_test, y_pred)
                    # Affichage des scores
                    st.write(f"Alpha utilisé pour l'évaluation: {alpha}")
                    st.write(f"Mean Squared Error: {mse_lasso}")
                    st.write(f"R^2 Score: {r2_lasso}")
                    # Affichage de la comparaison entre la prédiction et la target dans un expander
                    comparaison = pd.DataFrame({'Valeur Réelle (y_test)': y_test, 'Valeur Prédite (y_pred)': y_pred})
                    with st.expander("Comparaison entre la prédiction et la target"):
                        st.dataframe(comparaison)

                    # Séparation en colonne puis affichage des graphs
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
                            ax.plot(np.log10(alphas), coefs[i, :], label=X_train.columns[i])
                        ax.set_xlabel('Log10(Alpha)')
                        ax.set_ylabel('Coefficients')
                        ax.set_title('Chemin de Régularisation (Lasso Path)', loc='center')
                        ax.legend()
                        st.pyplot(fig)
        except ValueError:
            st.error("Veuillez entrer une valeur numérique valide pour alpha.")
    else:
        st.info("Cliquez sur 'Lancer GridSearchCV' pour commencer la recherche des hyperparamètres.")

# Fonction pour la régression lineaire
def show_linear_regression(X_train, X_test, y_train, y_test):
    st.subheader("Régression Linéaire")
    if st.button("Évaluer Régression Linéaire"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        comparaison = pd.DataFrame({
            'Valeur Réelle (y_test)': y_test,
            'Valeur Prédite (y_pred)': y_pred
        })
        # Calcul et affichage des scores
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"MSE: {mse}")
        st.write(f"R^2 Score: {r2}")
        with st.expander("Comparaison entre la prédiction et la target"):
            st.dataframe(comparaison)
        # Seperation en colonne puis affichage des graphs
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
            ax.set_title('Graphique de Dispersion', loc='center')
            ax.legend()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.regplot(x='Valeur Réelle (y_test)', y='Valeur Prédite (y_pred)', data=comparaison,
                        scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
            ax.set_xlabel('Valeurs Réelles (y_test)')
            ax.set_ylabel('Valeurs Prédites (y_pred)')
            ax.set_title('Graphique de Régression', loc='center')
            st.pyplot(fig)

# Fonction pour l'extra trees regressor
def show_extra_trees(X_train, X_test, y_train, y_test, random_state):
    st.subheader("Extra Trees Regressor")

    # Paramètres du modèle
    n_estimators = st.slider("Nombre d'arbres (n_estimators)", min_value=10, max_value=500, value=100, step=10, key="extra_trees_n_estimators")
    max_depth = st.slider("Profondeur maximale (max_depth)", min_value=1, max_value=40, value=10, key="extra_trees_max_depth")
    bootstrap = st.selectbox("Utiliser Bootstrap ?", [False, True], key="extra_trees_bootstrap")

    # Bouton pour lancer le modèle
    if st.button("Évaluer Extra Trees Regressor"):
        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcul ffichage des scores
        comparaison = pd.DataFrame({
            'Valeur Réelle (y_test)': y_test,
            'Valeur Prédite (y_pred)': y_pred
        })

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Évaluation du Modèle Extra Trees Regressor")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R^2 Score: {r2}")

        with st.expander("Comparaison entre la prédiction et la target"):
            st.dataframe(comparaison)

        # Seperation en colonne puis affichage des graphs
        col1, col2 = st.columns(2)

        with col1:
            fig = plot_learning_curve(model, X_train, y_train)
            st.pyplot(fig)

        with col2:
            feature_importances = model.feature_importances_
            features = X_train.columns
            fig, ax = plt.subplots()
            sns.barplot(x=feature_importances, y=features, ax=ax)
            ax.set_title('Importance des Caractéristiques')
            st.pyplot(fig)

# Fonction pour l'xgboost
def show_xgboost(X_train, X_test, y_train, y_test, random_state):
    st.subheader("XGBoost Regressor")

    n_estimators = st.slider("Nombre d'arbres (n_estimators)", min_value=10, max_value=500, value=100, step=10, key="xgboost_n_estimators")
    max_depth = st.slider("Profondeur maximale (max_depth)", min_value=1, max_value=20, value=6, key="xgboost_max_depth")
    learning_rate = st.slider("Taux d'apprentissage (learning_rate)", min_value=0.01, max_value=0.3, value=0.1, step=0.01, key="xgboost_learning_rate")
    gamma = st.slider("Gamma", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="xgboost_gamma")
    subsample = st.slider("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.1, key="xgboost_subsample")

    if st.button("Évaluer XGBoost Regressor"):
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                 gamma=gamma, subsample=subsample, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        comparaison = pd.DataFrame({
            'Valeur Réelle (y_test)': y_test,
            'Valeur Prédite (y_pred)': y_pred
        })

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R^2 Score: {r2}")

        with st.expander("Comparaison entre la prédiction et la target"):
            st.dataframe(comparaison)

        col1, col2 = st.columns(2)

        with col1:
            fig = plot_learning_curve(model, X_train, y_train)
            st.pyplot(fig)

        with col2:
            feature_importances = model.feature_importances_
            features = X_train.columns
            fig, ax = plt.subplots()
            sns.barplot(x=feature_importances, y=features, ax=ax)
            ax.set_title('Importance des Caractéristiques')
            st.pyplot(fig)


# Fonction principale pour afficher le Lazy Regressor
def show_lazy_regressor(X_train, X_test, y_train, y_test):
    st.caption("LazyRegressor")
    st.info("Le LazyRegressor permet de tester plusieurs modèles et identifier ceux qui s'adaptent le mieux à nos données.")

    # Bouton pour lancer LazyRegressor
    if st.button("Lancer LazyRegressor"):
        with st.spinner('Calcul des performances des modèles en cours...'):
            reg = LazyRegressor()
            models, predictions = reg.fit(X_train, X_test, y_train, y_test)

        st.success("Modèles évalués avec succès!")
        st.dataframe(models)

        # Interprétation basique des résultats
        best_model = models.index[0]
        st.write(
            f"Le modèle qui semble le plus performant est : **{best_model}** avec un coefficient de détermination (R²) de {models.loc[best_model, 'R-Squared']}.")

        # Explications des métriques
        st.subheader("Explication des métriques")
        st.write("""
            - **R² (Coefficient de détermination)** : Mesure la proportion de variance expliquée par le modèle. Un R² de 1 signifie que le modèle explique parfaitement les données, tandis qu'un R² de 0 indique que le modèle n'explique aucune variance.
            - **R² ajusté** : Comme le R², mais ajusté pour le nombre de prédicteurs dans le modèle. Il pénalise les modèles qui utilisent trop de variables non pertinentes.
            - **RMSE (Root Mean Squared Error)** : Indique la racine carrée de l'erreur quadratique moyenne. Plus la RMSE est faible, plus les prédictions du modèle sont proches des valeurs réelles.
        """)

        # Affichage sous forme de DataFrame des 3 meilleurs modèles
        st.subheader("Top 3 des meilleurs modèles :")
        top_3 = models.head(3)[['R-Squared', 'Adjusted R-Squared', 'RMSE', 'Time Taken']]
        st.dataframe(top_3)

        # Comparaison des 3 meilleurs modèles avec des graphiques dans deux colonnes
        st.subheader("Comparaison des 3 meilleurs modèles")

        # Création des colonnes
        col1, col2 = st.columns(2)

        # Graphique de comparaison des RMSE dans la première colonne
        with col1:
            fig1, ax1 = plt.subplots()
            sns.barplot(x=top_3.index, y=top_3['RMSE'], ax=ax1)
            for i, value in enumerate(top_3['RMSE']):
                ax1.text(i, value, round(value, 2), ha='center', va='bottom')
            ax1.set_title("Comparaison des RMSE")
            ax1.set_ylabel("RMSE")
            ax1.set_xlabel("Modèles")
            st.pyplot(fig1)

        # Graphique de comparaison des R² dans la deuxième colonne
        with col2:
            fig2, ax2 = plt.subplots()
            sns.barplot(x=top_3.index, y=top_3['R-Squared'], ax=ax2)
            for i, value in enumerate(top_3['R-Squared']):
                ax2.text(i, value, round(value, 4), ha='center', va='bottom')
            ax2.set_title("Comparaison des R²")
            ax2.set_ylabel("R²")
            ax2.set_xlabel("Modèles")
            st.pyplot(fig2)