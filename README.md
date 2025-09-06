# ❤️ Heart Disease Prediction using Machine Learning  

📌 **Table of Contents**  
- Project Overview  
- Dataset Description  
- Steps & Tasks Performed  
- Key Insights & Learnings  
- Project Deliverables  
- Live Demo  
- Tools & Technologies Used  
- Skills Demonstrated  
- Author  

---

## 📌 Project Overview  
This project predicts the likelihood of **heart disease** in patients based on clinical and lifestyle features such as blood pressure, cholesterol, BMI, and more.  
The solution applies **Machine Learning models** and deploys the best-performing model using a **Streamlit web app** for real-time predictions.  

The repository contains:  
- `Project.ipynb` → Jupyter notebook with ML workflow  
- `streamlit_app.py` → Streamlit web app source code  
- `model.pkl` → Trained ML model file  
- `heart-disease.csv` → Dataset used for training/testing  
- `requirements.txt` → Python dependencies  
- `Env/` → Conda environment folder (installed packages)  

---

## 📊 Dataset Description  
The dataset (`heart-disease.csv`) includes:  
- Age  
- Blood Pressure  
- Cholesterol  
- Blood Sugar  
- Resting ECG  
- Maximum Heart Rate Achieved  
- Exercise-Induced Angina  
- ST Depression & Slope  
- Target → `1` (disease present), `0` (no disease)  

---

## 🛠️ Steps & Tasks Performed  
🔹 **Data Preprocessing**  
- Handled missing values  
- Encoded categorical variables  
- Standardized numerical features  

🔹 **Exploratory Data Analysis (EDA)**  
- Distribution plots to understand feature behavior  
- Correlation heatmaps for identifying strong predictors  
- Feature importance analysis  

🔹 **Model Training**  
- Trained Logistic Regression, Random Forest, and XGBoost models  
- Compared performance using **Accuracy, ROC-AUC, and F1-score**  
- Tuned hyperparameters for better generalization  
- Saved best model (`model.pkl`)  

🔹 **Deployment**  
- Built interactive **Streamlit app**  
- Deployed on **Streamlit Cloud**  

---

## 📈 Key Insights & Learnings  
- **EDA is crucial**: Helped discover correlations, outliers, and important features influencing heart disease.  
- **Evaluation metrics matter**: Learned why metrics like **ROC-AUC** are more reliable than raw accuracy in healthcare settings.  
- **Hyperparameter tuning is essential**: Improved model performance significantly and prevented overfitting.  
- Logistic Regression was interpretable but less accurate, while Random Forest and XGBoost delivered stronger predictive power.  
- Final chosen model achieved **~89% accuracy** and **0.92 ROC-AUC score**.  

---

## 📂 Project Deliverables  
✅ Cleaned dataset (`heart-disease.csv`)  
✅ ML model file (`model.pkl`)  
✅ Jupyter Notebook (`Project.ipynb`)  
✅ Streamlit app (`streamlit_app.py`)  
✅ Environment & requirements files (`Env/`, `requirements.txt`)  

---

## 🌐 Live Demo  
🚀 Try the app here: [Heart Disease Prediction - Streamlit](https://heartdiseaseprediction-123.streamlit.app/)  

---

## 🚀 Tools & Technologies Used  
- **Python** (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost)  
- **Streamlit** for deployment  
- **Jupyter Notebook** for experimentation  
- **Conda** for environment management  

---

## 🧑‍💻 Skills Demonstrated  
- Data Preprocessing & Feature Engineering  
- **Importance of EDA for decision-making**  
- **Selection of correct evaluation metrics (ROC-AUC, F1, Accuracy)**  
- **Hyperparameter tuning for optimal performance**  
- Model Deployment with Streamlit  
- Version Control with Git & GitHub  

---

## 👤 Author  
**Mohammad Navaman Jamadar**  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/mohammad-navaman-jamadar)  
