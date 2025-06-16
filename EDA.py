import streamlit as st

# Load EDA Pkgs
import pandas as pd
import vaex as vx
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import datetime


# Load Data visulization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") # Establece el backend de matplotlib a 'Agg', que es un backend no interactivo
import seaborn as sns
import plotly.express as px # librería de visualización interactiva muy usada

def mostrar_estadisticas_categoricas(df):

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(categorical_cols) == 0:
        st.info("No se encontraron columnas categóricas en el dataset.")
        return

    for col in categorical_cols:
        if col == "customerid":
            continue
        else:
            st.markdown(f"### 🏷️ Columna: **{col}**")

            moda = df[col].mode()[0] if not df[col].mode().empty else "Sin datos"
            n_unique = df[col].nunique()

            st.write(f"🔹 **Moda**: {moda}")
            st.write(f"🔹 **Categorías únicas**: {n_unique}")

            # Tabla de frecuencias
            freq_abs = df[col].value_counts().rename("Frecuencia")
            freq_rel = df[col].value_counts(normalize=True).mul(100).round(2).rename("Porcentaje (%)")
            tabla = pd.concat([freq_abs, freq_rel], axis=1)

        st.dataframe(tabla)
def eda_app():
    st.subheader("Análisis Exploratorio de Datos")
    dataset = vx.open("./data/dataset_completo.parquet")
    dataset_partial_encoded = vx.open("./data/dataset_partial_encoded.parquet")
    dataset_pd = dataset.to_pandas_df()


    submenu = st.sidebar.selectbox("Opciones", ["Descriptivo", "Representación gráfica"])
    if submenu == "Descriptivo":

        st.dataframe(dataset_pd)
        dataset_pd["churn"] = dataset_pd["churn"].astype("category")

        with st.expander("Tipo de datos"):
            st.dataframe(dataset_pd.dtypes)

        with st.expander("Estadisticas básicas"):
            st.subheader("📊 Estadísticas descriptivas para variables numéricas")
            st.dataframe(dataset_pd.describe())

            st.subheader("📊 Estadísticas descriptivas para variables categóricas")
            mostrar_estadisticas_categoricas(dataset_pd)

        with st.expander("Valores nulos"):
            # Calcular los valores nulos por columna
            null_summary = {
                col: (dataset[col].isna()).sum().item()  # .item() para obtener el número de forma escalar
                for col in dataset.get_column_names()
                if (dataset[col].isna()).sum().item() > 0
            }

            if not null_summary:
                st.success("✅ No hay valores nulos en este dataset.")
            else:
                st.warning("Existen valores nulos en al menos una categoría")
                st.write("Estas son las columnas con valores nulos:")
                
                null_df = pd.DataFrame.from_dict(null_summary, orient='index', columns=["Total de nulos"])
                st.dataframe(null_df)


    elif submenu == "Representación gráfica":

        st.subheader("📈 Representación gráfica")
    
        # Detectar columnas numéricas usando Vaex
        numerical_features = [col for col in dataset.get_column_names() if dataset[col].dtype.kind in "iuf"]

        with st.expander("Detección de outliers en variables numéricas"):
            for column in numerical_features:
                # Convertir solo la columna actual a pandas para graficar
                df_plot = dataset[[column]].to_pandas_df()

                # Crear el violin plot con boxplot y puntos de outliers
                fig = px.violin(
                    df_plot,
                    y=column,
                    box=True,
                    points="outliers",
                    title=f"Distribución de {column}",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                fig.update_layout(
                    yaxis_title=column,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

        with st.expander("📅 Evolución diaria de churn (clientes que se dieron de baja)"):

            # Convertimos todo el dataset (o solo lo necesario) a pandas
            df_pd = dataset.to_pandas_df()

            # Asegurarnos que las fechas estén bien convertidas
            df_pd["churn_date"] = pd.to_datetime(df_pd["churn_date"], errors="coerce")

            # Filtrar clientes churn y fechas válidas
            churn_pd = df_pd[df_pd["churn"] == 1].copy()
            churn_pd = churn_pd.dropna(subset=["churn_date"])

            # Crear columna solo con el día
            churn_pd["churn_dia"] = churn_pd["churn_date"].dt.strftime("%Y-%m-%d")

            # Agrupar por día
            churn_diario = churn_pd.groupby("churn_dia").size().reset_index(name="n_churns")
            churn_diario["churn_dia"] = pd.to_datetime(churn_diario["churn_dia"])
            churn_diario = churn_diario.sort_values("churn_dia")

            # Calcular media móvil de 7 días
            churn_diario["churns_suavizados"] = churn_diario["n_churns"].rolling(window=7, center=True).mean()

            # Gráfico interactivo
            fig = px.line(
                churn_diario,
                x="churn_dia",
                y="churns_suavizados",
                title="Churn diario (media móvil 7 días)",
                labels={"churn_dia": "Día", "churns_suavizados": "Churn suavizado"},
            )

            fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Clientes dados de baja (media móvil)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Días en la empresa: clientes churn"):
            
            # Filtrar churners (clientes que se dieron de baja)
            churn_data = dataset[dataset["churn"] == 1]

            # Convertir solo las columnas necesarias a pandas para graficar
            df_plot = churn_data[["dias_de_contrato"]].to_pandas_df()

            # Crear el histograma
            fig = px.histogram(
                df_plot,
                x="dias_de_contrato",
                nbins=60,
                opacity=0.85,
                title="Distribución de días en la empresa (solo churners)",
                labels={"dias_de_contrato": "Días de contrato"},
                color_discrete_sequence=["#EF553B"]
            )

            fig.update_traces(marker_line_width=1.5, marker_line_color="white")
            fig.update_layout(
                xaxis_title="Días de contrato",
                yaxis_title="Número de clientes (churn)",
                font=dict(size=14),
                title_font=dict(size=16, family="Segoe UI", color="#ffffff"),
                plot_bgcolor="#1e1e1e",
                paper_bgcolor="#1e1e1e",
                font_color="#e0e0e0",
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
            )

            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Distribución de clientes churn por edad"):

            # Asegurar que birth_date está en formato datetime
            dataset_pd["birth_date"] = pd.to_datetime(dataset_pd["birth_date"], errors="coerce")

            # Calcular edad en años
            today = pd.Timestamp.today()
            dataset_pd["edad"] = (today - dataset_pd["birth_date"]).dt.days // 365

            # Filtrar columnas necesarias y eliminar nulos si hay
            df_plot = dataset_pd[["edad", "churn"]].dropna()

            # Crear gráfico violin
            fig = px.violin(
                df_plot,
                y="edad",
                x="churn",
                color="churn",
                box=True,
                title="Distribución de edad según si el cliente hizo churn",
                labels={"edad": "Edad", "churn": "Churn"},
                color_discrete_map={0: "#00CC96", 1: "#EF553B"}
            )

            # Mostrar en Streamlit
            st.plotly_chart(fig, use_container_width=True)


        with st.expander("📈 Correlación de variables numéricas y codificadas (binarias/ternarias) con churn"):

            # Solo columnas numéricas distintas de 'churn'
            numeric_cols = [
                col for col in dataset_partial_encoded.get_column_names()
                if dataset_partial_encoded[col].dtype.kind in "ifbu" and col != "churn"
            ]

            # Calcular correlaciones
            correlaciones = {
                col: dataset_partial_encoded.correlation(col, "churn")
                for col in numeric_cols
            }

            # Convertir a DataFrame ordenado por valor absoluto
            cor_df = (
                pd.DataFrame.from_dict(correlaciones, orient="index", columns=["correlación"])
                .sort_values(by="correlación", key=abs, ascending=False)
                .reset_index()
                .rename(columns={"index": "Variable"})
            )

            # Control deslizante para top-N
            top_n = st.slider("Selecciona cuántas variables mostrar", min_value=5, max_value=30, value=15)
            cor_df = cor_df.head(top_n)

            st.write(f"Top {top_n} variables más correlacionadas con churn:")

            cor_df["correlación"] = cor_df["correlación"].round(4)

            # Gráfico interactivo
            fig = px.bar(
                cor_df,
                x="correlación",
                y="Variable",
                orientation="h",
                title=f"Top {top_n}: Correlación (absoluta) con churn",
                color="correlación",
                color_continuous_scale="RdBu"
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Correlación",
                yaxis_title="Variable",
            )

            st.plotly_chart(fig, use_container_width=True)



