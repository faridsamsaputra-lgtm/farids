import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================================================
# APLIKASI PREDIKSI RISIKO DIABETES (FINAL – SESUAI ML)
# =====================================================

def prediction_app_diabetes():

    st.title("🩺 Sistem Prediksi Risiko Diabetes")
    st.write("Aplikasi ini menggunakan **Machine Learning (Logistic Regression)** untuk memprediksi risiko diabetes berdasarkan data medis pasien.")

    # =====================================================
    # LOAD MODEL & FEATURE
    # =====================================================
    try:
        model = joblib.load("model_diabetes.pkl")
        features = joblib.load("diabetes_features.pkl")
    except:
        st.error("❌ Model atau file fitur tidak ditemukan.")
        st.stop()

    # =====================================================
    # FORM INPUT
    # =====================================================
    with st.form("form_prediksi"):

        st.subheader("📋 Input Data Medis Pasien")

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Jumlah Kehamilan", 0, 20, 0)
            glucose = st.number_input("Kadar Glukosa (mg/dL)", 0, 300, 120)
            blood_pressure = st.number_input("Tekanan Darah (mmHg)", 0, 200, 70)
            skin_thickness = st.number_input("Ketebalan Kulit (mm)", 0, 100, 20)

        with col2:
            insulin = st.number_input("Kadar Insulin (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("BMI", 10.0, 70.0, 26.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Usia", 1, 120, 35)

        submit = st.form_submit_button("🔍 Prediksi")

    # =====================================================
    # PROSES PREDIKSI
    # =====================================================
    if submit:

        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])

        # pastikan urutan fitur sama persis
        input_data = input_data[features]

        # ===============================
        # PREDIKSI MODEL
        # ===============================
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.write("---")
        st.subheader("📊 Hasil Prediksi Machine Learning")

        st.metric("Probabilitas Diabetes", f"{probability*100:.2f}%")

        # =====================================================
        # INTERPRETASI BERDASARKAN OUTPUT MODEL
        # =====================================================

        if prediction == 1:
            st.error("🚨 MODEL MEMPERKIRAKAN PASIEN MENGALAMI DIABETES")
            status = "POSITIF DIABETES"
        else:
            st.success("✅ MODEL MEMPERKIRAKAN PASIEN TIDAK DIABETES")
            status = "NEGATIF DIABETES"

        st.write(f"**Hasil Klasifikasi:** {status}")

        # =====================================================
        # INTERPRETASI ILMIAH
        # =====================================================
        st.subheader("📘 Interpretasi Model")

        st.write(
            f"Berdasarkan hasil pemodelan menggunakan algoritma Logistic Regression, "
            f"pasien diprediksi berada pada kondisi **{status}** dengan probabilitas "
            f"sebesar **{probability*100:.2f}%**. "
            f"Nilai probabilitas ini menunjukkan tingkat keyakinan model terhadap prediksi yang dihasilkan."
        )

        # =====================================================
        # ANALISIS FAKTOR DOMINAN
        # =====================================================
        st.subheader("⚠️ Faktor Risiko yang Mempengaruhi")

        faktor = []

        if glucose >= 126:
            faktor.append("Glukosa darah tinggi (>126 mg/dL)")
        if bmi >= 25:
            faktor.append("BMI berlebih / obesitas")
        if age >= 45:
            faktor.append("Usia berisiko")
        if blood_pressure >= 90:
            faktor.append("Tekanan darah tinggi")
        if dpf >= 0.8:
            faktor.append("Riwayat genetik diabetes")

        if faktor:
            for f in faktor:
                st.write(f"• {f}")
        else:
            st.write("Tidak terdapat faktor risiko dominan berdasarkan input.")

        # =====================================================
        # REKOMENDASI
        # =====================================================
        st.subheader("🩺 Rekomendasi")

        rekomendasi = []

        if glucose >= 126:
            rekomendasi.append("Mengontrol asupan gula dan karbohidrat")
        if bmi >= 25:
            rekomendasi.append("Menurunkan berat badan secara bertahap")
        if blood_pressure >= 90:
            rekomendasi.append("Rutin mengontrol tekanan darah")
        if prediction == 1:
            rekomendasi.append("Disarankan melakukan pemeriksaan medis lanjutan")

        for r in rekomendasi:
            st.write(f"✔ {r}")

        st.warning("Hasil prediksi ini bersifat pendukung keputusan dan tidak menggantikan diagnosis dokter.")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    prediction_app_diabetes()
