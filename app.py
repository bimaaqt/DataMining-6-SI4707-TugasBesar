import streamlit as st
import joblib
import numpy as np

# Load model dan preprocessing
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
le_smoke = joblib.load('le_smoke.pkl')

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")

st.title("🔍 Prediksi Risiko Penyakit Jantung")
st.write("Masukkan informasi berikut untuk melihat prediksi:")

# Input user
age = st.number_input("Umur", 25, 90, 50)
sleep_hours = st.slider("Jam Tidur per Hari", 0.0, 12.0, 7.0)
gender = st.selectbox("Jenis Kelamin", le_gender.classes_)
cholesterol = st.number_input("Kadar Kolesterol", 100, 350, 200)
smoking_status = st.selectbox("Status Merokok", le_smoke.classes_)

# Prediksi
if st.button("Prediksi"):
    user_data = np.array([[age, sleep_hours, gender, cholesterol, smoking_status]])
    user_data[:, 2] = le_gender.transform(user_data[:, 2])
    user_data[:, 4] = le_smoke.transform(user_data[:, 4])
    user_data = scaler.transform(user_data.astype(float))
    
    prediction = model.predict(user_data)
    prob = model.predict_proba(user_data)[0][1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ Risiko tinggi terkena penyakit jantung ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Risiko rendah terkena penyakit jantung ({prob*100:.2f}%)")
