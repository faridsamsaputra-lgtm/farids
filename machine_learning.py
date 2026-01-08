import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE 
import joblib

def ml_model_diabetes():
    # 1. Membaca Dataset
    df = pd.read_csv('diabetes (2).csv')
    
    st.title("Analisis Prediksi Diabetes (Logistic Regression)")

    # --- BAGIAN HASIL PENANGANAN MISSING VALUES ---
    st.write('### 1. Hasil Penanganan Missing Values (Nilai 0 Medis)')
    cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Menghitung jumlah nol sebelum diperbaiki untuk ditampilkan
    null_counts = {col: int((df[col] == 0).sum()) for col in cols_fix}
    
    for col in cols_fix:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    # Menampilkan tabel hasil missing values
    df_missing = pd.DataFrame(list(null_counts.items()), columns=['Fitur Medis', 'Jumlah Nilai 0 (Awal)'])
    st.table(df_missing)
    st.success("Semua nilai 0 di atas telah berhasil digantikan dengan nilai Median.")

    # --- BAGIAN HASIL DETEKSI OUTLIER ---
    st.write('### 2. Hasil Deteksi Outlier (Metode IQR)') 
    numbers = df.drop(columns=['Outcome']).columns
    
    Q1 = df[numbers].quantile(0.25)
    Q3 = df[numbers].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    old_shape = df.shape[0]
    df = df[~((df[numbers] < lower_bound) | (df[numbers] > upper_bound)).any(axis=1)]
    new_shape = df.shape[0]
    
    # Menampilkan ringkasan angka outlier
    col_out1, col_out2, col_out3 = st.columns(3)
    col_out1.metric("Data Awal", f"{old_shape} baris")
    col_out2.metric("Outlier Dibuang", f"{old_shape - new_shape} baris")
    col_out3.metric("Data Bersih", f"{new_shape} baris")

    st.write('**Preview Dataset Bersih:**')
    st.dataframe(df.head())

    # --- BAGIAN VISUALISASI PERBANDINGAN ---
    st.write('### 3. Visualisasi Perbandingan (Density Plot)')
    st.write("Grafik di bawah menunjukkan perbandingan distribusi data sebelum dan sesudah normalisasi (Min-Max Scaling).")
    
    df_select = df.copy()
    scaler = MinMaxScaler()
    df_select[numbers] = scaler.fit_transform(df_select[numbers])

    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.write('**Sebelum Normalisasi (Glucose)**')
        chart_pre = alt.Chart(df).transform_density('Glucose', as_=['Glucose', 'density']).mark_area(opacity=0.5).encode(
            x=alt.X("Glucose:Q", title="Skala Asli (0-200)"), 
            y="density:Q").properties(width=350, height=200)
        st.altair_chart(chart_pre, use_container_width=True)

    with col_vis2:
        st.write('**Setelah Normalisasi (Glucose)**')
        chart_post = alt.Chart(df_select).transform_density('Glucose', as_=['Glucose', 'density']).mark_area(opacity=0.5, color='orange').encode(
            x=alt.X("Glucose:Q", title="Skala Normal (0-1)"), 
            y="density:Q").properties(width=350, height=200)
        st.altair_chart(chart_post, use_container_width=True)

    # 4. Correlation Heatmap
    st.write('### 4. Korelasi Antar Variabel')
    corr = df.corr().reset_index().melt('index')
    corr.columns = ['Variable1', 'Variable2', 'Correlation']
    heatmap = alt.Chart(corr).mark_rect().encode(
        x='Variable2:N', y='Variable1:N',
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Variable1', 'Variable2', 'Correlation']
    ).properties(width=600, height=400)
    st.altair_chart(heatmap, use_container_width=True)

    # 5. Handling Imbalance (SMOTE)
    st.write("### 5. Handling Imbalance & Split Data")
    X = df_select.drop("Outcome", axis=1)
    y = df_select["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    st.write(f"Distribusi Kelas Setelah SMOTE: **Sehat (0): {(y_train_res==0).sum()}, Diabetes (1): {(y_train_res==1).sum()}**")

    # 6. Pemodelan & Interpretasi
    st.write('### 6. Pemodelan & Interpretasi Risiko')
    model = LogisticRegression()
    model.fit(X_train_res, y_train_res)
    
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient (β)": model.coef_[0]
    })
    coef_df["Odds Ratio (Risk)"] = np.exp(coef_df["Coefficient (β)"])
    coef_df = coef_df.sort_values(by="Coefficient (β)", ascending=False)
    
    col_res1, col_res2 = st.columns([6,4])
    with col_res1:
        st.write("**Tabel Koefisien & Odds Ratio:**")
        st.dataframe(coef_df)
    with col_res2:
        st.write("**Kesimpulan Faktor:**")
        top_risk = coef_df.iloc[0]
        st.success(f"Faktor risiko utama adalah **{top_risk['Feature']}**.")
        st.info(f"Setiap kenaikan 1 satuan fitur ini meningkatkan risiko sebesar **{(top_risk['Odds Ratio (Risk)']-1)*100:.1f}%**.")

    # 7. Evaluasi
    st.write('### 7. Evaluasi Model Akhir')
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        cm = confusion_matrix(y_test, y_pred)
        st.write("**Confusion Matrix:**")
        st.write(cm)
        
    with col_ev2:
        st.metric("Akurasi", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
        st.metric("Recall", f"{recall_score(y_test, y_pred)*100:.2f}%")
        st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba)*100:.2f}%")

    # Simpan Model
    joblib.dump(model, "model_diabetes.pkl")
    joblib.dump(X.columns, "diabetes_features.pkl")
def ml_model():
    ml_model_diabetes()

if __name__ == "__main__":
    ml_model_diabetes()