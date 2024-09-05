import streamlit as st
import pandas as pd
import io


def nettoyageData_page():
    st.title("Nettoyage des données 🎈")

    # Vérification si des données sont présentes
    if 'df' in st.session_state:
        # Récupération de la base de données enregistrer
        df = st.session_state['df']

        # Afficher le nombre de lignes et de colonnes
        st.subheader("Taille de votre base de données")

        st.write(f"Nombre de lignes : {df.shape[0]}")
        st.write(f"Nombre de colonnes : {df.shape[1]}")

        # Afficher les informations du DataFrame sous forme de tableau
        st.subheader("Informations sur votre base de données")

        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()

        # Parse the df.info() output to extract useful information
        info_data = []
        for line in s.split("\n")[5:-3]:  # Skipping the header and summary
            parts = line.split()
            # Nom de la colonne, Nombre de valeurs non nulles, Type de données
            info_data.append({
                "Nombre de données non_null": parts[-3],
                "Type de données": parts[-1],
                "Quantité des données manquantes": round(((df.shape[0] - int(parts[-3]))/ df.shape[0]) * 100, 3)
            })

        # Create a DataFrame from the extracted info
        df_info = pd.DataFrame(info_data)
        # Add the column names from df.columns as a separate column
        df_info["Nom de la variable"] = df.columns
        # Reorder the columns to match the desired order
        df_info = df_info[["Nom de la variable", "Nombre de données non_null", "Type de données", "Quantité des données manquantes"]]

        # Ajout d'une colonne pour la suppression des données
        df_sup = df_info.copy()
        df_sup['Suppression'] = False

        # Display the DataFrame
        df_sup = st.data_editor(
            df_sup,
            column_config={
                "Nom de la variable": st.column_config.TextColumn(
                    "Nom de la variable",
                    help="Nom de la variable",
                    default="st.",
                    max_chars=50,
                    validate=r"^st\.[a-z_]+$",
                ),
                "Nombre de données non_null": st.column_config.TextColumn(
                    "Nombre de données non_null",
                    help="Nombre de données non_null",
                    default="st.",
                    max_chars=50,
                    validate=r"^st\.[a-z_]+$",
                ),
                "Type de données": st.column_config.TextColumn(
                    "Type de données",
                    help="Type de données",
                    default="st.",
                    max_chars=50,
                    validate=r"^st\.[a-z_]+$",
                ),
                "Quantité des données manquantes": st.column_config.ProgressColumn(
                    "Quantité des données manquantes",
                    help="Pourcentage des données manquantes",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Suppression": st.column_config.CheckboxColumn(
                    "Supprimer la colonne?",
                    help="Si vous cocher la cellule, la colonne sera supprimer",
                    default=False,
            )
            },
            hide_index=True,
        )

        st.write("Vous pouvez supprimer et n'oubliez pas d'enregistrer vos modifications ! ")

        # Suppression des colonnes cochées
        column_to_drop = df_sup["Nom de la variable"][df_sup["Suppression"] == True]
        if len(column_to_drop) > 0:
            df = df.drop(columns=column_to_drop)
            st.markdown("**Aperçu des données :**")
            st.write(df.head(5))
            if st.button("Enregistrer"):
                st.session_state['df'] = df
                st.success("Données enregistrées avec succès!")

        # Etudes de la typologie des données
        st.subheader("Typologie des données")
        st.write("Vérifiez que toutes vos données sont de type numérique pour éviter la non-compatibilité avec certains modèles de machine learning.")

        if st.checkbox("Consulter la typologie des données"):
            st.write("Vérifiez que la colonne 'Numérique' soit coché pour chaque variable")
            # Capture info from df into a buffer and extract useful data
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()

            # Parse the df.info() output to extract useful information
            info_data = []
            for line in s.split("\n")[5:-3]:  # Skipping the header and summary
                parts = line.split()
                # Nom de la colonne, Nombre de valeurs non nulles, Type de données
                info_data.append({
                    "Type de données": parts[-1]
                })

            # Create a DataFrame from the extracted info
            df_typologie = pd.DataFrame(info_data)
            # Add the column names from df.columns as a separate column
            df_typologie["Nom de la variable"] = df.columns
            # Vérifier si la colonne est de type numérique en utilisant df
            df_typologie["Numérique"] = df_typologie["Nom de la variable"].apply(lambda col: pd.api.types.is_numeric_dtype(df[col]))

            # Reorder the columns to match the desired order
            df_typologie = df_typologie[["Nom de la variable", "Type de données", "Numérique"]]

            # Affichage des informations dans un tableau (data_editor)
            st.data_editor(
                df_typologie,
                column_config={
                    "Nom de la variable": st.column_config.TextColumn(
                        "Nom de la variable",
                        help="Nom de la variable",
                    ),
                    "Type de données": st.column_config.TextColumn(
                        "Type de données",
                        help="Type de données pour chaque colonne",
                    ),
                    "Numérique": st.column_config.CheckboxColumn(
                        "Numérique",
                        help="La variable est-elle de type numérique?",
                        disabled=True  # Empêche l'utilisateur de modifier la colonne
                    ),
                },
                hide_index=True,
            )

        #############################################################

        # Détection des données manquantes
        st.subheader("Données manquantes")
        missing_data = df.isnull().sum()
        st.write(f"La base de données est composée de {missing_data.values.sum()} données manquantes")
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            missing_data_df = pd.DataFrame({
                "Nom de la variable": missing_data.index,
                "Nombre de données manquantes": missing_data.values
            })
            st.write(missing_data_df)
        else:
            st.write("Aucune donnée manquante détectée.")

        # Détection des données non numériques
        st.subheader("Données non numériques")
        categorical_columns = df.select_dtypes(exclude=['number']).columns
        if not categorical_columns.empty:
            categorical_data_df = pd.DataFrame({
                "Nom de la variable": categorical_columns,
                "Type de données": df[categorical_columns].dtypes.values
            })
            st.write(categorical_data_df)
        else:
            st.write("Aucune donnée non numérique détectée.")

        # Suppression des données
        st.subheader("Suppression des colonnes")
        columns = df.columns.tolist()
        column_to_drop = st.multiselect("Sélectionnez les colonnes à supprimer", columns)
        if len(column_to_drop) > 0:
            data = df.drop(columns=column_to_drop)
            st.write(f"Données après suppression des colonnes {column_to_drop}:")
            st.write(data.head())

        # Options de nettoyage
        st.subheader("Imputation des données manquantes")
        if st.checkbox("Supprimer les lignes avec des valeurs manquantes"):
            data = df.dropna()
            st.write("Données après suppression des valeurs manquantes :")
            st.write(data.head())

    else:
        st.write("Aucune donnée n'a été chargée. Veuillez charger les données via la page Source de données.")