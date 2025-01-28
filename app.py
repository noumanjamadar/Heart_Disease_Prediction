import streamlit as st
import pandas as pd
import pickle


# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# Streamlit app
st.title("Heart Disease Prediction App")
st.write(
    "Enter the values for the features to predict the likelihood of heart disease."
)

# Input fields for user
st.write("### Enter Input Features")
user_input = {
    "age": st.number_input("Age", min_value=0, max_value=120, value=30),
    "sex": st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1]),
    "cp": st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3]),
    "trestbps": st.number_input("Resting Blood Pressure", min_value=0, value=120),
    "chol": st.number_input("Cholesterol", min_value=0, value=200),
    "fbs": st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1]
    ),
    "restecg": st.selectbox("Resting ECG Results (0-2)", [0, 1, 2]),
    "thalach": st.number_input("Max Heart Rate Achieved", min_value=0, value=150),
    "exang": st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1]),
    "oldpeak": st.number_input(
        "ST Depression Induced by Exercise", value=0.0, format="%.2f"
    ),
    "slope": st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2]),
    "ca": st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4]),
    "thal": st.selectbox("Thalassemia (1-3)", [1, 2, 3]),
}

# Convert input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Prediction
if st.button("Predict"):
    prediction = model.predict(user_input_df)[0]
    result = (
        "Person is suffering from  Heart Disease"
        if prediction == 1
        else "Person is not suffering from  Heart Disease"
    )
    st.write(f"Prediction: *{result}*")
