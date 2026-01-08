import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def prediction_app_diabetes():
    st.title("Sistem Prediksi Risiko Diabetes")
    st.write("Masukkan data medis pasien untuk memprediksi risiko diabetes menggunakan model Logistic Regression.")
    
    # 1. Load Model & Metadata
    # Menggunakan try-except untuk menangani jika file model belum di-generate
    try:
        model = joblib.load("model_diabetes.pkl")
        feature_names = joblib.load("diabetes_features.pkl")
    except FileNotFoundError:
        st.error("⚠️ File 'model_diabetes.pkl' tidak ditemukan. Harap jalankan tab 'Machine Learning' terlebih dahulu untuk melatih model.")
        return

    # 2. Form Input Pengguna (Disesuaikan dengan diabetes (2).csv)
    st.write("### Input Data Medis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan", 0, 20, 0)
    with col2:
        glucose = st.number_input("Kadar Glukosa (mg/dL)", 0, 300, 100)
    with col3:
        blood_pressure = st.number_input("Tekanan Darah (mm Hg)", 0, 150, 70)
    with col4:
        skin_thickness = st.number_input("Ketebalan Kulit (mm)", 0, 100, 20)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        insulin = st.number_input("Kadar Insulin (mu U/ml)", 0, 900, 80)
    with col2:
        bmi = st.number_input("BMI (Indeks Massa Tubuh)", 0.0, 70.0, 25.0)
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    with col4:
        age = st.number_input("Usia (Tahun)", 1, 120, 30)

    # 3. Logika Kategori Otomatis (Sama seperti referensi dashboard Anda)
    st.write("---")
    col1, col2, col3 = st.columns(3)

    # Age Category
    with col1:
        if age <= 30:
            age_cat, color = "Young Adult", "#4CAF50"
        elif age <= 50:
            age_cat, color = "Middle Aged", "#EBD300"
        else:
            age_cat, color = "Senior", "#F44336"
        st.markdown(f"""<div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                    <div style="font-size:13px;">Kategori Usia</div>
                    <div style="font-size:16px;font-weight:bold;">{age_cat}</div></div>""", unsafe_allow_html=True)

    # BMI Category
    with col2:
        if bmi < 18.5:
            bmi_cat, color = "Underweight", "#4DA6FF"
        elif bmi < 25:
            bmi_cat, color = "Normal", "#4CAF50"
        elif bmi < 30:
            bmi_cat, color = "Overweight", "#EBD300"
        else:
            bmi_cat, color = "Obesity", "#F44336"
        st.markdown(f"""<div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                    <div style="font-size:13px;">Kategori BMI</div>
                    <div style="font-size:16px;font-weight:bold;">{bmi_cat}</div></div>""", unsafe_allow_html=True)

    # Glucose Category
    with col3:
        if glucose < 100:
            glu_cat, color = "Normal", "#4CAF50"
        elif glucose < 126:
            glu_cat, color = "Prediabetes", "#EBD300"
        else:
            glu_cat, color = "Diabetes", "#F44336"
        st.markdown(f"""<div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                    <div style="font-size:13px;">Kategori Glukosa</div>
                    <div style="font-size:16px;font-weight:bold;">{glu_cat}</div></div>""", unsafe_allow_html=True)

    st.write("") 

    # 4. Pemrosesan Data untuk Prediksi
    user_df = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    # Penting: Urutan fitur harus sama dengan saat training model
    user_processed = user_df[feature_names]

    # 5. Eksekusi Prediksi
    if st.button("Analisis Risiko Diabetes"):
        # Hitung Probabilitas (Logistic Regression)
        prob = model.predict_proba(user_processed)[0][1]
        prob_percent = prob * 100
        
        st.write("### 🔍 Hasil Analisis")
        st.metric("Tingkat Keyakinan Risiko", f"{prob_percent:.2f}%")

        if prob >= 0.5:
            st.error("⚠️ Hasil: Pasien berisiko tinggi menderita **Diabetes**.")
            st.markdown("""
            **Rekomendasi Tindakan:**
            - Segera lakukan konsultasi dengan tenaga medis profesional.
            - Pantau asupan karbohidrat dan gula secara ketat.
            - Lakukan pemeriksaan Glukosa Darah Puasa (GDP).
            """)
        else:
            st.success("✅ Hasil: Pasien berada pada risiko **Rendah** (Sehat).")
            st.markdown("""
            **Rekomendasi Tindakan:**
            - Pertahankan pola makan seimbang dan gaya hidup aktif.
            - Lakukan pemeriksaan kesehatan rutin tahunan.
            """)

        with st.expander("Lihat Detail Teknis"):
            st.write(f"Raw Probability Score: `{prob:.4f}`")
            st.write("Threshold klasifikasi yang digunakan adalah 0.50.")

# Jalankan fungsi jika file dieksekusi secara langsung
if __name__ == "__main__":
    prediction_app_diabetes()