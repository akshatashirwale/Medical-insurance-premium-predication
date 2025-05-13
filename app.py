import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Insurance Charge Predictor")
st.title("Insurance Charges Prediction App")
st.markdown("Enter customer details to predict medical insurance charges.")

# Input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Feature encoding
input_data = {
    'age': age,
    'bmi': bmi,
    'smoker_yes': 1 if smoker == 'yes' else 0,
    
}

input_df = pd.DataFrame([input_data])

if st.button("Predict Charges"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Insurance Charges: {prediction:.2f}")