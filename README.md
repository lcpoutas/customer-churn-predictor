# 📉 Customer Churn Prediction App

Esta aplicación permite predecir la probabilidad de abandono de clientes (`churn`) mediante un modelo de regresión logística entrenado con `SGDClassifier`. El usuario puede explorar los datos y realizar predicciones personalizadas a través de una interfaz interactiva construida con **Streamlit**.

---

## 📁 Estructura del Proyecto

```
CustomerChurn_App/
├── app.py                         # Punto de entrada de la app Streamlit
├── prediction.py                  # Módulo para predicción personalizada
├── EDA.py                         # Módulo para el análisis exploratorio
├── requirements.txt               # Librerías necesarias
├── Dockerfile                     # Archivo para la construcción del contenedor
├── data/
│   ├── billing.csv
│   ├── clients.parquet
│   ├── tenure.json
│   ├── dataset_completo.parquet
│   ├── dataset_encoded.parquet
│   ├── prepared_dataset.parquet
│   ├── modelo_final_SGD.pkl
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── cols_to_scale.pkl
│   ├── columnas_entrenamiento.pkl
│   ├── final_columns.pkl
│   └── umbral_optimo_SGD.txt
└── Notebooks/                     # Jupyter Notebooks de desarrollo
```

---

## 🚀 Cómo Ejecutar el Proyecto

### ✅ Opción 1: Con Docker (Recomendado)

1. **Construir la imagen Docker:**
```bash
docker build -t customer-churn-app .
```

2. **Ejecutar la app en el puerto 8502:**
```bash
docker run -p 8502:8502 customer-churn-app streamlit run app.py --server.port=8502 --server.address=0.0.0.0
```

3. **Abrir en el navegador:**
[http://localhost:8502](http://localhost:8502)

---

### ⚙️ Opción 2: Ejecutar Localmente (sin Docker)

> Requiere Python 3.9+ y un entorno virtual

1. **Crear y activar entorno:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

3. **Lanzar la app:**
```bash
streamlit run app.py
```

---

## 🧠 Funcionalidades Principales

- **Exploración de datos (EDA)**: Gráficas, distribución de variables, análisis de churn por fecha, etc.
- **Predicción personalizada**: Sección interactiva para introducir datos de un cliente y obtener predicción.
- **Modelo robusto**:
  - Entrenado con SGDClassifier y GridSearchCV
  - Regularización L1 / L2 / ElasticNet
  - Optimización del umbral de decisión (F1-score)
- **Preprocesamiento reproducible**:
  - Imputación de valores nulos
  - One-Hot Encoding de variables categóricas
  - Escalado de variables continuas

---

## 🧾 Requisitos

Revisa el archivo [`requirements.txt`](./requirements.txt) para ver todas las dependencias, incluyendo:

- `streamlit`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `vaex` (opcional para lectura rápida de datos)

---

## 📌 Notas Técnicas

- El modelo ha sido entrenado sobre un dataset de más de 318.000 registros.
- El umbral de decisión fue ajustado a `0.465` tras analizar el F1-score.
- Es imprescindible mantener los archivos `.pkl` y `.txt` del directorio `/data` para que la app funcione correctamente.

---

## 👨‍💻 Autor

Desarrollado como parte del curso de **Machine Learning** – Grado en Inteligencia Artificial  
Contacto: Luis Carlos de Vicente Poutás – lcpoutas@gmail.com
