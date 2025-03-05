import joblib
import numpy as np
import streamlit as st

# Memuat model tanpa normalisasi menggunakan joblib
model_terbaik = joblib.load("adaboost_best_model.pkl")

# Fungsi prediksi
def predict(model, data):
    prediction = model.predict(data)
    return prediction[0]

# Streamlit UI
st.title("Prediksi Diabetes")

st.header("Krisdova Rio Alvonsa 210411100165")

# Input data
st.header("Masukkan Data")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Umur", min_value=0, step=1)

with col2:
    HbA1c_levelc = st.number_input("HbA1c", min_value=0.0, step=0.1)
    blood_glucose_level = st.number_input("Gula Darah", min_value=0.0, step=0.1)

# Konversi jenis kelamin menjadi angka
gender_numerik = 0 if gender == "Perempuan" else 1

# Data baru
new_data = np.array([[gender_numerik, age, HbA1c_levelc, blood_glucose_level]])

# Tombol prediksi
if st.button("Prediksi"):
    result = predict(model_terbaik, new_data)
    
    # Tampilkan hasil
    st.header("Hasil Prediksi")
    if result == 0:
        st.success("Pasien diprediksi memiliki diabetes.")
    else:
        st.success("Pasien diprediksi tidak memiliki diabetes.")
