# ============================================================
# 🍔 APLIKASI PREDIKSI KEPUASAN PELANGGAN
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import os

# ============================================================
# 1️⃣ Load Model
# ============================================================
@st.cache_resource
def load_model():
    """
    Memuat model Pipeline yang sudah terlatih dari file pkl.
    """
    model_path = "model_kepuasan_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan.")
        st.error("Mohon jalankan skrip pelatihan di Colab untuk membuat file ini.")
        st.stop()
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        st.error(f"❌ Gagal memuat aset: {e}. Pastikan versi pustaka Anda konsisten.")
        st.stop()

# Muat model saat aplikasi dimulai
model_pipeline = load_model()

# ============================================================
# 2️⃣ App Configuration
# ============================================================
st.set_page_config(
    page_title="Prediksi Kepuasan Pelanggan",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Aplikasi Prediksi Kepuasan Pelanggan")
st.markdown("Masukkan detail pelanggan untuk memprediksi apakah mereka puas ('Yes') atau tidak ('No').")

# ============================================================
# 3️⃣ Input Form
# ============================================================
with st.form("satisfaction_prediction_form"):
    st.subheader("📝 Masukkan Detail Pelanggan")
    
    # Menentukan pilihan untuk kolom kategorikal
    gender_options = ["Male", "Female"]
    marital_status_options = ["Single", "Married", "Prefer not to say"]
    occupation_options = ["Student", "Employee", "Self Employed", "Housewife"]
    monthly_income_options = ["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"]
    edu_options = ["Graduate", "Post Graduate", "Ph.D", "School", "Uneducated"]
    feedback_options = ["Positive", "Negative "]

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia", min_value=15, max_value=100, value=25)
        gender = st.selectbox("Jenis Kelamin", gender_options)
        marital_status = st.selectbox("Status Pernikahan", marital_status_options)
        occupation = st.selectbox("Pekerjaan", occupation_options)
        monthly_income = st.selectbox("Pendapatan Bulanan", monthly_income_options)
    
    with col2:
        edu_qual = st.selectbox("Kualifikasi Pendidikan", edu_options)
        family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3)
        latitude = st.number_input("Latitude", value=12.97, format="%.4f")
        longitude = st.number_input("Longitude", value=77.59, format="%.4f")
        feedback = st.selectbox("Feedback (Input)", feedback_options)

    submitted = st.form_submit_button("🔮 Prediksi Kepuasan")
    
    if submitted:
        try:
            # Buat DataFrame dari input pengguna
            input_data = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'Marital Status': marital_status,
                'Occupation': occupation,
                'Monthly Income': monthly_income,
                'Educational Qualifications': edu_qual,
                'Family size': family_size,
                'latitude': latitude,
                'longitude': longitude,
                'Feedback': feedback
            }])
            
            # Melakukan prediksi menggunakan pipeline
            prediction = model_pipeline.predict(input_data)
            
            # Mengonversi hasil prediksi kembali ke label 'Yes'/'No'
            predicted_label = "Yes" if prediction[0] == 1 else "No"
            
            st.subheader("✅ Prediksi Berhasil!")
            if predicted_label == "Yes":
                st.success(f"Berdasarkan data yang dimasukkan, pelanggan ini **cenderung puas**.")
            else:
                st.warning(f"Berdasarkan data yang dimasukkan, pelanggan ini **cenderung tidak puas**.")
            st.info(f"Hasil prediksi model: **{predicted_label}**")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
            st.warning("Mohon periksa kembali input Anda.")