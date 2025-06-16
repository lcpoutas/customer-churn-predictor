```markdown
# 📉 Customer Churn Predictor App

This application predicts the probability of customer churn using various Machine Learning models. Built with **Streamlit**, it allows users to interactively explore the data, select a model, and make real-time predictions.

---

## 🧠 Key Features

✅ Predict churn with **9 different models**:

- Random Forest (Ensemble)
- LightGBM (Ensemble)
- XGBoost (Ensemble)
- KNN (Ensemble)
- SGDClassifier (Optimized)
- MLP (Neural Network)
- SVM
- Naive Bayes (Bernoulli, Multinomial, Gaussian)

📊 Visual and interactive data exploration (EDA)

🧪 Reproducible preprocessing pipeline: encoding, scaling, and column selection

🚀 Compatible with Docker and local deployment

---

## 📁 Project Structure

```
CustomerChurn_App/
├── app.py             # Main Streamlit entry point
├── prediction.py      # Prediction logic and model selection
├── EDA.py             # Exploratory data analysis interface
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker container configuration
│
├── models/            # Trained models (.pkl and .h5)
│                     # (recommend using Git LFS)
│
├── data/              # Encoders, scalers, feature sets and processed datasets
│
└── Notebooks/         # Jupyter notebooks for development and experimentation
```

---

## 🛠️ Technologies Used

- `scikit-learn`, `imbalanced-learn`, `scikeras`
- `tensorflow`, `xgboost`, `lightgbm`
- `vaex`, `pandas`, `numpy`
- `streamlit`, `plotly`, `seaborn`, `matplotlib`
- `joblib`, `pyarrow`
- `Docker`, `Git LFS`

---

## 🚀 How to Run the App

### Option A: Docker (Recommended)

```bash
# 1. Build the Docker image
docker build -t churn-predictor-app .

# 2. Run the app
docker run -p 8502:8502 churn-predictor-app streamlit run app.py --server.port=8502 --server.address=0.0.0.0
```

Open in browser: [http://localhost:8502](http://localhost:8502)

---

### Option B: Run Locally (Without Docker)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ⚠️ Important Notes

- This project uses **Git LFS** to manage large model files (>100MB).
  - Install from: [https://git-lfs.github.com](https://git-lfs.github.com)
  - Then run: `git lfs install`

- If you hit the LFS quota limit, consider storing models externally:
  - Google Drive, Hugging Face Hub, or any cloud storage

---

## 📊 Model F1-Scores (Churn Class)

| Model                    | F1 Score |
|--------------------------|----------|
| 🎯 Random Forest         | 0.994    |
| KNN                      | 0.953    |
| LightGBM                 | 0.944    |
| MLP (Neural Network)     | 0.923    |
| XGBoost                  | 0.828    |
| SVM                      | 0.808    |
| SGDClassifier            | 0.749    |
| Naive Bayes (Bernoulli)  | 0.326    |
| Naive Bayes (Multinomial)| 0.304    |
| Naive Bayes (Gaussian)   | 0.230    |

---

## 👨‍💻 Author

Developed by **Luis Carlos de Vicente Poutás**  
🎓 Project for *Machine Learning* – Artificial Intelligence Bachelor's Degree  
📫 Contact: `lcpoutas@gmail.com`
```
