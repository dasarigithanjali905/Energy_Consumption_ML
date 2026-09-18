# ⚡ Energy Consumption Prediction & AI Analytics

An ML-powered web application for **energy consumption prediction, model analytics, ticket classification, and AI-generated responses**.

The application is built using **Python, Scikit-learn, XGBoost, Pandas, Plotly, Joblib, Streamlit, and Hugging Face APIs**.

---

## 🚀 Features

* 📊 Energy Consumption Dashboard
* 🤖 Machine Learning based energy prediction
* 📈 Model performance and analytics
* 🔍 Accuracy and confusion-matrix visualization
* 🎯 Feature importance visualization
* 🧠 Ticket classification
* 💬 AI-generated customer support responses
* 📁 CSV dataset upload
* 🔄 Model retraining from the Admin Panel
* ⚡ Interactive Streamlit web interface

---

## 🧠 Machine Learning

The application automatically determines whether the target variable represents a **classification or regression** problem.

For regression tasks, the project supports:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

For classification tasks, the project supports:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

The models are trained and evaluated, and the model with the highest calculated score is selected as the best model. For regression, the stored score represents **R²**; for classification, it represents **accuracy**.

The training pipeline saves the trained model, preprocessing pipeline, and evaluation metrics for later predictions.

---

## 📊 Dataset

The configured target column is:


EnergyConsumption

The configured input columns are:

Temperature
Humidity
SquareFootage
Occupancy
HVACUsage
LightingUsage
RenewableEnergy
DayOfWeek
Holiday

The preprocessing pipeline automatically identifies numeric, categorical, and text columns and applies appropriate preprocessing techniques.

## Numeric features use median imputation, categorical features can use one-hot or label encoding, and text features can be transformed using TF-IDF.

## 🏗️ Project Structure

Energy_Consumption/
│
├── app.py
├── config.py
├── hf_api.py
├── predict.py
├── train.py
├── utils.py
├── requirements.txt
├── README.md
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
│
└── assets/
    └── style.css


### Main Files

| File               | Description                              |
| ------------------ | ---------------------------------------- |
| `app.py`           | Streamlit application and user interface |
| `config.py`        | Project paths and configuration          |
| `train.py`         | Model training and evaluation            |
| `predict.py`       | Prediction functions                     |
| `utils.py`         | Dataset loading and preprocessing        |
| `hf_api.py`        | Hugging Face LLM API integration         |
| `requirements.txt` | Python dependencies                      |
| `data/dataset.csv` | Training dataset                         |
| `models/`          | Saved ML model, preprocessor and metrics |

## The Streamlit application contains separate pages for Dashboard, Model Prediction, Ticket Classifier, AI Response Generator, Model Analytics, and Admin Panel.

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Plotly
* Streamlit
* Joblib
* Requests
* Hugging Face API

The dependencies are listed in `requirements.txt`.


## 🔧 Installation

### 1. Clone the repository

bash
git clone https://github.com/YOUR_USERNAME/Energy_Consumption.git
cd Energy_Consumption


### 2. Create a virtual environment

Windows:

bash
python -m venv venv
venv\Scripts\activate

Linux/macOS:

bash
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies
bash
pip install -r requirements.txt


## 📁 Dataset

Place your CSV dataset inside:

data/dataset.csv

The dataset should contain the configured target column:

EnergyConsumption


and the required input features.


## 🤖 Train the Model

Run:
bash
python train.py


The training process creates:

models/model.pkl
models/preprocessor.pkl
models/metrics.json


The application also checks for the saved model and preprocessor when starting and can train them if they do not exist.


## ▶️ Run the Application

Start the Streamlit application:

bash
streamlit run app.py

Then open the Streamlit URL shown in your terminal, normally:

http://localhost:8501


## 📈 Model Prediction

The **Model Prediction** page accepts the building/environment features and uses the saved preprocessing pipeline and ML model to generate predicted energy consumption.

The prediction module loads the saved model and preprocessor before transforming the input and generating predictions.


## 📊 Model Analytics

The Model Analytics page provides visualizations including:

* Model accuracy / R² comparison
* Confusion matrix for classification
* Feature importance when supported by the selected model

The metrics are stored in:

models/metrics.json

and are displayed through interactive Plotly charts.


## 💬 AI Response Generator

The application integrates with the Hugging Face API to generate professional responses.

The application expects a Hugging Face access token through the environment variable:

HF_TOKEN

The API integration sends requests to the Hugging Face chat-completions endpoint and uses the configured DeepSeek model.

### Windows

bash
set HF_TOKEN=your_huggingface_token
streamlit run app.py


### Linux/macOS

bash
export HF_TOKEN=your_huggingface_token
streamlit run app.py

## 🔄 Admin Panel

The Admin Panel allows users to:

1. Upload a CSV dataset.
2. Save the uploaded dataset.
3. Retrain the ML model.

The Streamlit application provides CSV upload and model retraining functionality directly through the interface.


## 🔐 Security

Do not commit API keys, passwords, tokens, or other secrets to GitHub.

Create a `.gitignore` file containing:

gitignore
# Python
__pycache__/
*.py[cod]
venv/
.env

# Model files
models/*.pkl
models/*.joblib

# Local datasets
data/*.csv

# Secrets
.env
.streamlit/secrets.toml


If your dataset and trained model are intentionally part of your public repository, you can remove the corresponding `data/*.csv` and `models/*.pkl` entries.


## 🧪 Example Workflow

CSV Dataset
     │
     ▼
Data Preprocessing
     │
     ├── Numeric Features
     ├── Categorical Features
     └── Text Features
     │
     ▼
Feature Transformation
     │
     ▼
ML Model Training
     │
     ├── Linear Regression
     ├── Random Forest
     └── XGBoost
     │
     ▼
Best Model Selection
     │
     ▼
Saved Model + Preprocessor
     │
     ▼
Streamlit Application
     │
     ├── Prediction
     ├── Analytics
     ├── Ticket Classification
     └── AI Response Generation


## 📌 Future Improvements

* Add train/test split and cross-validation
* Add additional regression metrics such as MAE and RMSE
* Add hyperparameter tuning
* Add model versioning
* Add authentication for the Admin Panel
* Add real-time energy monitoring
* Add downloadable prediction reports
* Improve explainability with SHAP
* Deploy the application to a cloud platform


