import streamlit as st

# Load ML Pkgs
import joblib
import os

# Load EDA Pkgs
import numpy as np
import pandas as pd


attrib_info = """
- **Age**: Edad del cliente en años
- **Gender**: Género del cliente (Female, Male)
- **Monthly Charges**: Importe mensual cobrado al cliente
- **Internet Service**: Tipo de servicio de Internet (No, DSL, Fiber optic)
- **Online Security**: Servicio de seguridad online (Yes, No)
- **Online Backup**: Servicio de respaldo online (Yes, No)
- **Device Protection**: Protección del dispositivo (Yes, No)
- **Tech Support**: Soporte técnico (Yes, No)
- **Phone Lines**: Número de líneas telefónicas
- **Streaming**: Servicio de streaming contratado (Yes, No)
- **Contract Type**: Duración del contrato (Month-to-month, One year, Two year)
- **Married**: Estado civil (Yes, No)
- **Children**: Número de hijos
- **Paperless Billing**: Factura electrónica (Yes, No)
- **Payment Method**: Método de pago (Electronic check, Mailed check, Bank transfer, Credit card)
"""

label_dict = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}
internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}

def run_ml_app():
    st.subheader("🔍 Predicción de baja de cliente (churn)")
    
    with st.expander("ℹ️ Información de los atributos"):
        st.markdown(attrib_info)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad", 18, 100)
        gender = st.radio("Género", list(gender_map.keys()))
        monthly_charges = st.number_input("Cargo mensual", min_value=0.0)
        internet = st.selectbox("Servicio de Internet", list(internet_map.keys()))
        online_security = st.radio("Seguridad online", list(label_dict.keys()))
        online_backup = st.radio("Backup online", list(label_dict.keys()))  # NUEVO
        device_protection = st.radio("Protección de dispositivo", list(label_dict.keys()))
        phone_lines = st.slider("Líneas de teléfono", 0, 5)
        dias_de_contrato = st.number_input("Dias de contrato")

    with col2:
        contract = st.radio("Tipo de contrato", list(contract_map.keys()))
        married = st.radio("¿Está casado/a?", list(label_dict.keys()))
        children = st.slider("Número de hijos", 0, 10)
        tech_support = st.radio("Soporte técnico", list(label_dict.keys()))
        streaming = st.radio("Streaming contratado", list(label_dict.keys()))
        paperless = st.radio("Factura electrónica", list(label_dict.keys()))
        payment_method = st.selectbox("Método de pago", list(payment_map.keys()))
        country = st.selectbox("País", ["ES", "PT"])  # NUEVO
        contract_channel = st.selectbox("Canal de contratación", ["Store", "Phone", "Internet"])  # NUEVO

    # Muestra la selección
    with st.expander("📋 Tus datos introducidos"):
        input_dict = {
            "age": age,
            "gender": gender,
            "monthlycharges": monthly_charges,
            "internetservice": internet,
            "onlinesecurity": online_security,
            "onlinebackup": online_backup,  # NUEVO
            "deviceprotection": device_protection,
            "phone_lines": phone_lines,
            "contract": contract,
            "married": married,
            "children": children,
            "techsupport": tech_support,
            "streaming": streaming,
            "paperlessbilling": paperless,
            "paymentmethod": payment_method,
            "country": country,  # NUEVO
            "contract_channel": contract_channel,  # NUEVO
            "dias_de_contrato": dias_de_contrato
        }
        st.write(input_dict)

    with st.expander("Predicción del resultado"):

        # Menú desplegable para elegir el modelo a usar, mostrando el F1-score sobre clase 1 (churn)
        modelo_seleccionado = st.selectbox("Selecciona el modelo para predecir:", [
            "Random Forest (F1: 0.994)",
            "KNN (F1: 0.953)",
            "LightGBM (F1: 0.944)",
            "MLP (Red Neuronal) (F1: 0.923)",
            "XGBoost (F1: 0.828)",
            "SVM (F1: 0.808)",
            "SGDClassifier (F1: 0.749)",
            "Naive Bayes - Bernoulli (F1: 0.326)",
            "Naive Bayes - Multinomial (F1: 0.304)",
            "Naive Bayes - Gaussian (F1: 0.230)"
        ])

        # Al pulsar el botón, se lanza el flujo de predicción
        if st.button("📈 Predecir"):

            # 1. Convertir el diccionario con la entrada del usuario a un DataFrame
            input_df = pd.DataFrame([input_dict])

            # 2. Cargar el encoder de variables categóricas
            encoder = joblib.load("./data/encoder.pkl")

            # 3. Definir las variables categóricas utilizadas durante el entrenamiento
            categorical_cols = [
                'internetservice', 'onlinesecurity', 'onlinebackup',
                'deviceprotection', 'techsupport', 'streaming', 'gender',
                'paperlessbilling', 'paymentmethod', 'married',
                'country', 'contract_channel'
            ]

            # 4. Variables numéricas usadas directamente
            numeric_cols = ['monthlycharges', 'phone_lines', 'children', 'dias_de_contrato']

            # 5. Aplicar codificación a las variables categóricas
            encoded_array = encoder.transform(input_df[categorical_cols])
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

            # 6. Combinar variables codificadas y numéricas
            input_full = pd.concat([encoded_df, input_df[numeric_cols].reset_index(drop=True)], axis=1)

            # 7. Cargar el orden y nombres esperados por el modelo
            columnas_modelo = joblib.load("./data/final_columns.pkl")

            # 8. Rellenar columnas faltantes con 0 (por si el encoder omite alguna en un nuevo dato)
            input_full = input_full.reindex(columns=columnas_modelo, fill_value=0)

            # 9. Cargar el escalador y columnas a escalar
            scaler = joblib.load("./data/scaler.pkl")
            cols_to_scale = joblib.load("./data/cols_to_scale.pkl")

            # 10. Escalar sólo las columnas numéricas necesarias
            input_full[cols_to_scale] = scaler.transform(input_full[cols_to_scale])

            # 11. Preparar ruta del modelo basado en nombre limpio (sin F1)
            from tensorflow.keras.models import load_model
            modelo_limpio = modelo_seleccionado.split(" (")[0]  # Ej: "Random Forest"

            modelo_paths = {
                "Random Forest": "./models/random_forest_ensemble_models.pkl",
                "KNN": "./models/knn_ensemble_models.pkl",
                "LightGBM": "./models/lightgbm_ensemble_models.pkl",
                "MLP (Red Neuronal)": "./models/mlp_model.h5",
                "XGBoost": "./models/xgboost_ensemble_models.pkl",
                "SVM": "./models/svm_ensemble_models.pkl",
                "SGDClassifier": "./models/modelo_final_SGD.pkl",
                "Naive Bayes - Bernoulli": "./models/naive_bayes_ensemble_bernoulli.pkl",
                "Naive Bayes - Multinomial": "./models/naive_bayes_ensemble_multinomial.pkl",
                "Naive Bayes - Gaussian": "./models/naive_bayes_ensemble_gaussian.pkl"
            }

            modelo_path = modelo_paths[modelo_limpio]

            # 12. Cargar modelo: si es .h5 es una red neuronal, si no, un modelo sklearn
            if modelo_path.endswith(".h5"):
                # --- Red neuronal MLP ---
                modelo = load_model(modelo_path)
                prob = modelo.predict(input_full)[0][0]
                umbral = 0.5

            else:
                modelo = joblib.load(modelo_path)

                # Si es una lista de modelos (ensemble), agregamos probabilidades
                if isinstance(modelo, list):
                    prob = np.mean([m.predict_proba(input_full)[:, 1][0] for m in modelo])
                    umbral = 0.5  # puedes personalizarlo si quieres
                else:
                    # Si es SGDClassifier, aplicamos umbral personalizado
                    if modelo_limpio == "SGDClassifier":
                        try:
                            with open("./data/umbral_optimo_SGD.txt", "r") as f:
                                umbral = float(f.read())
                        except:
                            umbral = 0.5
                    else:
                        umbral = 0.5

                    prob = modelo.predict_proba(input_full)[:, 1][0]

            # 13. Clasificar según umbral
            pred = int(prob >= umbral)

            # 14. Mostrar resultado en pantalla
            st.write(f"🔮 **Probabilidad de churn con `{modelo_limpio}`:** {prob:.2%}")
            if pred == 1:
                st.error("⚠️ El modelo predice que este cliente probablemente abandonará el servicio.")
            else:
                st.success("✅ El modelo predice que este cliente probablemente permanecerá.")




