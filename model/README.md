# ✈️ Turbofan Engine RUL Predictor

This Streamlit application predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines using machine learning models trained on NASA's C-MAPSS dataset. It features explainability tools (SHAP), model comparison, and multiple input modes for real-time and batch predictions.

For a full step by step review of Project stages, Please review [4_Model_Report](CMAPSSData/notebooks/4_Model_Report.ipynb)

---

## 📊 Features

- **LSTM Deep Learning** model for accurate RUL prediction  
- **Interactive Streamlit Dashboard** with clean UI  
- **Upload your own data** or use preloaded sample engines  
- **Batch engine evaluation**  
- **SHAP Explainability**: Understand which sensors impact predictions  
- **Model Comparison**: LSTM, Random Forest, and XGBoost performance  
- 🔐 Optional login functionality with `streamlit-authenticator`

---

## 📂 Project Structure

### turbofan-rul-prediction/
### CMAPSSData/
- ### App/
    - #### assets/
        - hytec-illustration.png
    - #### model/
        - keras_lstm_tuned.h5
        - scaler.pkl
    - app.py
    - config.toml
    - hash_test.py
    - logs.csv
    - model_metrics.json
    - multi_engine_test_input.csv
    - notebook.ipynb
    - sample_input.csv
    - sample_shap_values.csv
- ### README.md
- ### requirements.txt
- ### .gitignore


---

## 📈 How It Works

1. Upload or select turbofan engine input data (30-timestep window).
2. The model scales inputs and feeds them to a pre-trained **LSTM neural network**.
3. Predictions are displayed instantly with download options.
4. Explore model explainability using **SHAP** plots.
5. Compare other model performances with bar charts of MAE/RMSE.

---

## 🧪 Tech Stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [SHAP](https://github.com/slundberg/shap)
- [Plotly](https://plotly.com/)
- [scikit-learn](https://scikit-learn.org/)
- [joblib](https://joblib.readthedocs.io/)

---

## 🚀 Getting Started Locally

### 🔧 1. Clone the Repo

```bash
git clone https://github.com/Mu111Ghoz/turbofan-rul-prediction.git
cd turbofan-rul-prediction
```
### 📦 2. Create & Activate Conda Environment
```bash
conda create -n aircraft_dashboard python=3.10
conda activate aircraft_dashboard
```
### 📥 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### ▶️ 4. Run the App
```bash
cd App
streamlit run app.py
```
### 📚 NASA C-MAPSS Dataset
The model is trained on [NASA's CMAPSS datasets](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), which simulate engine degradation over time with multiple operational settings and sensor measurements.
### 📄 License
This project is released under the MIT License.
### 🤝 Acknowledgements
- NASA Prognostics Center of Excellence
- SHAP Library by Scott Lundberg
- Streamlit for simplifying ML dashboards
### 👨‍💻 Author
Muhamed Ghoz
- GitHub [Muhamed Ghoz](https://github.com/Mu111Ghoz)
- LinkedIn [Muhamed Ghoz](www.linkedin.com/in/muhamed-abdelfattah-ghoz-38636184)