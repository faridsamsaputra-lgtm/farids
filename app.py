import streamlit as st
import pandas as pd
import numpy as np

# =========================================
# KONFIGURASI HALAMAN
# =========================================
st.set_page_config(
    page_title="Analisis Komparatif Algoritma Klasifikasi Machine Learning untuk Prediksi Risiko Diabetes",
    layout="wide"
)

st.header("Analisis Prediksi Risiko Diabetes")
st.write("**UAS BIOSTATISTIKA 1.0** - Universitas Muhammadiyah Semarang")
st.write("DIBUAT OLEH : FARID SAM SAPUTRA [ B2D023020 ]")
st.write("Semarang, 19 Januari 2026")
st.write("SUMBER DATA : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")

# =========================================
# TABS
# =========================================
tab1, tab2, tab7, tab3, tab4, tab6 = st.tabs([
    "About Dataset",
    "Dashboards",
    "Langkah Processing",
    "Machine Learning",
    "Prediction App",
    "Contact Me"
])

# =========================================
# TAB 1 - ABOUT
# =========================================
with tab1:
    import about
    about.about_dataset()

# =========================================
# TAB 2 - DASHBOARD
# =========================================
with tab2:
    import visualisasi
    visualisasi.chart()

# =========================================
# TAB 3 - PREPROCESSING
# =========================================
with tab7:
    st.subheader("⚙️ Langkah Processing Data Diabetes")

    aktifkan = st.checkbox("Aktifkan Processing Data")
    jalankan = st.button("Jalankan Processing")

    if not aktifkan:
        st.info("Centang terlebih dahulu untuk menampilkan proses.")

    elif aktifkan and not jalankan:
        st.warning("Klik tombol Jalankan Processing.")

    elif aktifkan and jalankan:
        try:
            df = pd.read_csv("diabetes (2).csv")

            # 1 LOAD DATA
            st.markdown("### 1️⃣ Load Dataset")
            st.dataframe(df.head(), use_container_width=True)

            # 2 DATA CLEANING
            st.markdown("### 2️⃣ Data Cleaning & Validasi")
            st.text(df.info())
            st.dataframe(df.isnull().sum())

            # 3 STATISTIK
            st.markdown("### 3️⃣ Statistik Deskriptif")
            st.dataframe(df.describe(), use_container_width=True)

            # 4 EDA
            st.markdown("### 4️⃣ Exploratory Data Analysis (EDA)")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.bar_chart(df["Glucose"])

            with col2:
                st.bar_chart(df["BMI"])

            with col3:
                st.bar_chart(df["Age"])

            # 5 HANDLING MISSING VALUES
            st.markdown("### 5️⃣ Penanganan Missing Values (Nilai 0 Medis)")
            kolom_medis = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

            df_mv = df.copy()
            df_mv[kolom_medis] = df_mv[kolom_medis].replace(0, np.nan)

            st.dataframe(df_mv.isnull().sum())

            df_mv[kolom_medis] = df_mv[kolom_medis].fillna(df_mv[kolom_medis].median())
            st.success("Missing values berhasil ditangani dengan median")

            # 6 OUTLIER IQR
            st.markdown("### 6️⃣ Deteksi Outlier (Metode IQR)")
            Q1 = df_mv.quantile(0.25)
            Q3 = df_mv.quantile(0.75)
            IQR = Q3 - Q1

            outlier = ((df_mv < (Q1 - 1.5 * IQR)) | (df_mv > (Q3 + 1.5 * IQR))).sum()
            st.dataframe(outlier)

            st.info("Outlier tidak dihapus untuk menjaga kestabilan data")

            # 7 VISUALISASI DISTRIBUSI
            st.markdown("### 7️⃣ Visualisasi Perbandingan Distribusi")
            st.line_chart(df_mv[["Glucose", "BMI", "Age"]])

            # 8 IMBALANCE DATA + SPLIT
            st.markdown("### 8️⃣ Handling Imbalance & Split Data")

            from sklearn.model_selection import train_test_split
            from imblearn.over_sampling import SMOTE

            X = df_mv.drop("Outcome", axis=1)
            y = df_mv["Outcome"]

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)

            st.write("Distribusi kelas setelah SMOTE:")
            st.dataframe(pd.Series(y_res).value_counts())

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42
            )

            st.success(
                "✔ Missing values tertangani\n"
                "✔ Outlier terdeteksi\n"
                "✔ Distribusi divisualisasikan\n"
                "✔ Data imbalance diperbaiki\n"
                "✔ Data siap training dan testing"
            )

        except FileNotFoundError:
            st.error("File diabetes (2).csv tidak ditemukan")

# =========================================
# TAB 4 - MACHINE LEARNING
# =========================================
with tab3:
    import machine_learning
    machine_learning.ml_model()

# =========================================
# TAB 5 - PREDICTION
# =========================================
with tab4:
    import prediction
    prediction.prediction_app_diabetes()

# =========================================
# TAB 6 - CONTACT
# =========================================
with tab6:
    st.markdown(
        "**Nama** : FARID SAM SAPUTRA  \n"
        "**Email** : faridsamsaputra@gmail.com  \n"
        "**Lokasi** : Semarang, Jawa Tengah"
    )
