import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib


def ml_model():

    st.subheader("📘 Machine Learning - Analisis Diabetes")

    # ===============================
    # LOAD DATA
    # ===============================
    df = pd.read_csv("diabetes (2).csv")

    # ===============================
    # PREPROCESSING MEDIS
    # ===============================
    cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_fix:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # ===============================
    # SPLIT DATA
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===============================
    # NORMALISASI (UNTUK MODEL)
    # ===============================
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===============================
    # HANDLE IMBALANCE
    # ===============================
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(
        X_train_scaled, y_train
    )

    # ===============================
    # MODEL
    # ===============================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # ======================================================
    # 1️⃣ PEMODELAN & INTERPRETASI RISIKO (DATA ASLI)
    # ======================================================
    st.markdown("## 1️⃣ Pemodelan & Interpretasi Risiko")

    coef_df = pd.DataFrame({
        "Fitur": X.columns,
        "Koefisien (β)": model.coef_[0]
    })

    coef_df["Odds Ratio"] = np.exp(coef_df["Koefisien (β)"])
    coef_df = coef_df.sort_values("Odds Ratio", ascending=False)

    def kategori_risiko(or_val):
        if or_val >= 2:
            return "🔴 Risiko Tinggi"
        elif or_val >= 1.3:
            return "🟠 Risiko Sedang"
        else:
            return "🟢 Risiko Rendah"

    coef_df["Kategori Risiko"] = coef_df["Odds Ratio"].apply(kategori_risiko)

    st.dataframe(coef_df, use_container_width=True)

    faktor = coef_df.iloc[0]
    st.success(
        f"Faktor risiko dominan adalah **{faktor['Fitur']}** "
        f"dengan Odds Ratio **{faktor['Odds Ratio']:.2f}**."
    )

    # ======================================================
    # 2️⃣ EVALUASI MODEL (PLOT)
    # ======================================================
    st.markdown("## 2️⃣ Evaluasi Model Akhir")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col2.metric("Precision", f"{precision_score(y_test, y_pred)*100:.2f}%")
    col3.metric("Recall", f"{recall_score(y_test, y_pred)*100:.2f}%")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred)*100:.2f}%")

    # ===============================
    # CONFUSION MATRIX PLOT
    # ===============================
    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

    st.pyplot(fig)

    # ===============================
    # ROC CURVE
    # ===============================
    st.markdown("### ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    st.pyplot(fig2)

    # ======================================================
    # 3️⃣ PEMILIHAN MODEL TERBAIK
    # ======================================================
    st.markdown("## 3️⃣ Pemilihan Model Terbaik")

    st.write("""
    Model **Logistic Regression** dipilih sebagai model terbaik karena:
    - Memiliki interpretasi medis yang jelas
    - Menyediakan Odds Ratio
    - Stabil pada dataset kesehatan
    - Cocok untuk deteksi dini penyakit
    """)

    # ======================================================
    # 4️⃣ AKURASI MODEL
    # ======================================================
    st.markdown("## 4️⃣ Akurasi Model")

    st.success(
        f"Model menghasilkan akurasi sebesar "
        f"**{accuracy_score(y_test, y_pred)*100:.2f}%** "
        f"dan memiliki kemampuan klasifikasi yang baik."
    )

    # ===============================
    # SIMPAN MODEL
    # ===============================
    joblib.dump(model, "model_diabetes.pkl")
    joblib.dump(X.columns, "diabetes_features.pkl")
