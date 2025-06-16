import streamlit as st
import streamlit.components.v1 as stc


# Import main functions app
from EDA import eda_app
from prediction import run_ml_app


html_template = """
    <style>
        .header-container {
            background: linear-gradient(90deg, #a18cd1 0%, #fbc2eb 100%);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            margin-bottom: 25px;
        }
        .header-title {
            color: white;
            font-size: 38px;
            font-weight: 700;
            margin: 0;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .header-subtitle {
            color: #f8f8f8;
            font-size: 18px;
            text-align: center;
            margin-top: 10px;
            font-family: 'Segoe UI', sans-serif;
        }
        .icon {
            font-size: 48px;
            text-align: center;
            margin-bottom: 12px;
        }
    </style>

    <div class="header-container">
        <div class="icon">📱📉</div>
        <h1 class="header-title">Predicción de Churn de Clientes</h1>
        <p class="header-subtitle">Una herramienta interactiva para analizar y predecir la pérdida de clientes en telecomunicaciones</p>
    </div>
"""
def main():

    menu =["Inicio", "EDA", "Prediccion"]

    choice = st.sidebar.selectbox("Menu", menu)

    stc.html(html_template, height=310)

    if choice == "Inicio":
        st.subheader("Inicio")
        st.markdown("""
            ### 📱 Customer Churn Predictor App

            Esta aplicación permite analizar y predecir la probabilidad de que un cliente abandone su compañía de telecomunicaciones (churn) en los próximos 100 días. Utiliza algoritmos de machine learning y técnicas de análisis de datos para apoyar estrategias de retención.

            #### 🗃️ Fuente de datos
            - Datos proporcionados en el campus virtual de la asignatura *Fundamentos de la Ciencia de Datos*.

            #### 🧠 Contenido de la App

            - **EDA (Análisis Exploratorio de Datos)**:  
            Visualización interactiva y análisis para entender las características de los clientes y los factores que pueden influir en el churn.

            - **ML (Modelo de Machine Learning)**:  
            Predicción basada en modelos de regresión logística. Permite estimar la probabilidad de que un cliente se convierta en churner, a partir de sus características.
            """)

    elif choice == "EDA":
        eda_app()
    elif choice == "Prediccion":
        run_ml_app()
    else:
        st.subheader("About")

if __name__ == "__main__":
    main()