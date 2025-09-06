# ❤️ Heart Disease Prediction Model

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Accuracy](https://img.shields.io/badge/Accuracy-89%25-brightgreen)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.92-yellowgreen)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LogReg%20%7C%20RandomForest-orange)

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Project Deliverables](#-project-deliverables)
- [Quick Run Instructions](#-quick-run-instructions)
- [Key Learnings](#-key-learnings)
- [Results](#-results)
- [Streamlit App](#-streamlit-app)
- [Tools & Technologies Used](#-tools--technologies-used)
- [Skills Demonstrated](#-skills-demonstrated)
- [Connect with Me](#-connect-with-me)

---

## 🚀 Project Overview

This project demonstrates how Machine Learning can assist in early detection of heart disease, which is critical in saving lives and reducing healthcare costs.  

The model predicts the likelihood of heart disease based on clinical features such as blood pressure, cholesterol, age, BMI, and sugar levels, helping in early diagnosis and preventive care.

---

## 📂 Dataset

**Source:** Cleveland dataset from the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).  

Includes clinical attributes such as:  
- Age, Sex  
- Resting Blood Pressure  
- Cholesterol levels  
- Fasting Blood Sugar  
- ECG results  
- Maximum Heart Rate Achieved  
- Exercise Induced Angina  
- Oldpeak, Slope, Thalassemia, etc.

---

## 🛠 Project Workflow

**Exploratory Data Analysis (EDA) 🔍**  
- Identified correlations between features.  
- Visualized patterns in patients with and without heart disease.  

**Data Preprocessing 🧹**  
- Handled missing values.  
- Encoded categorical variables.  
- Scaled numerical features.  

**Model Training 🤖**  
- Logistic Regression  
- Random Forest Classifier  

**Evaluation Metrics 📊**  
- Confusion Matrix  
- Precision, Recall, F1-Score  
- ROC-AUC  

**Deployment 🌐**  
- Built an interactive Streamlit App for real-time predictions.

---

## 📦 Project Deliverables

This repository contains everything needed to run and deploy the project:  
- Dataset CSV file 📂 (used for training and testing)  
- Trained Model File (model.pkl) 🤖  
- Streamlit Application Script (app.py) 🌐  
- Environment File (env/) ⚙️ containing all required Python libraries  
- Jupyter Notebook (Project.ipynb) 📒 with EDA, preprocessing, training, and evaluation

---

## ⚡ Quick Run Instructions

For technical reviewers who want to test the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Key Learnings  
- Importance of **EDA** in understanding feature distribution and spotting outliers.  
- **Selection of Evaluation Metrics** (Precision, Recall, F1-score, ROC-AUC) is critical in healthcare, where false negatives can be dangerous.  
- **Hyperparameter Tuning** improves model generalization and reduces overfitting.  
- End-to-end pipeline: from raw data → model → **deployment** on Streamlit.  

---

## 📈 Results  
- Achieved **89% Accuracy** ✅  
- ROC-AUC Score: **0.92** 🏆  
- Reliable predictions with balanced **Precision & Recall** 
# Visualizations:

<table>
<tr>
<td><img src="Project%20Images/confusion_matrix.png" alt="Confusion Matrix" width="250"/></td>
<td><img src="Project%20Images/roc_auc_curve.png" alt="ROC AUC Curve" width="250"/></td>
<td><img src="Project%20Images/train_test_score_plot.png" alt="Train/Test Score Plot" width="250"/></td>
<td><img src="Project%20Images/Heart_pred_app.png" alt="Streamlit App Screenshot" width="250"/></td>
</tr>
<tr>
<td align="center">Confusion Matrix</td>
<td align="center">ROC AUC Curve</td>
<td align="center">Train/Test Score Plot</td>
<td align="center">Streamlit App Screenshot</td>
</tr>
</table>


---

## 🌐 Streamlit App  
Try the live interactive app here 👉 [Heart Disease Prediction App](https://heartdiseaseprediction-123.streamlit.app/)  

---

## 🛠 Tools & Technologies Used  
- **Python** 🐍  
- **Pandas, NumPy** for data handling  
- **Matplotlib, Seaborn** for EDA & Visualization  
- **Scikit-learn** for ML modeling  
- **Streamlit** for deployment  

---

## 🧑‍💻 Skills Demonstrated  
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Training & Evaluation  
- Hyperparameter Tuning  
- Deployment with Streamlit  

---

## 🤝 Connect with Me  
📌 GitHub: [Check out my GitHub Profile for other projects](https://github.com/noumanjamadar/)  
🌐 Live App: [Streamlit App](https://heartdiseaseprediction-123.streamlit.app/)  
💼 LinkedIn: [Mohammad Navaman Jamadar](https://www.linkedin.com/in/mohammad-navaman-jamadar)  
