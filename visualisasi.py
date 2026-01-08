import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def chart_diabetes():
    # 1. Membaca Dataset Diabetes
    df = pd.read_csv('diabetes (2).csv')
    
    # Pre-processing: Mengganti nilai 0 yang tidak logis dengan Median (seperti saran medis)
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    # Menambahkan Kolom Kategori untuk Visualisasi (mirip referensi stroke)
    df['bmi_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], 
                                labels=['Underweight', 'Normal', 'Overweight', 'Obesity'])
    df['age_category'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], 
                                labels=['Young Adult', 'Middle Aged', 'Senior'])
    
    # Judul
    st.title("Dashboard Monitoring Pasien Diabetes")

    # Hitung Metrik Utama
    total_pasien = df.shape[0]
    total_diabetes = df['Outcome'].sum()
    diabetes_rate = (total_diabetes / total_pasien) * 100

    # 2. Card Metrics dan Button Filter
    col1, col2, col3, col4, col5, col6 = st.columns([2,2,3,2,2,1])
    with col1:
        st.metric(label="Total Pasien", value=total_pasien)
    with col2:
        st.metric(label="Pasien Diabetes", value=total_diabetes)
    with col3:
        st.metric(label="Prevalensi", value=f"{diabetes_rate:.2f}%")
    
    # Initialize session state untuk filter
    if 'selected_age_cat' not in st.session_state:
        st.session_state.selected_age_cat = None
    if 'selected_outcome' not in st.session_state:
        st.session_state.selected_outcome = None
    
    with col4:
        st.write('**Kategori Usia**')
        if st.button("Young Adult"):
            st.session_state.selected_age_cat = 'Young Adult'
        if st.button("Senior"):
            st.session_state.selected_age_cat = 'Senior'
    with col5:
        st.write('**Status**')
        if st.button("Diabetes"):
            st.session_state.selected_outcome = 1
        if st.button("Sehat"):
            st.session_state.selected_outcome = 0
    with col6:
        if st.button("🔄"):
            st.session_state.selected_age_cat = None
            st.session_state.selected_outcome = None
            st.rerun()
    
    # Apply filter
    filtered_df = df.copy()
    if st.session_state.selected_age_cat:
        filtered_df = filtered_df[filtered_df['age_category'] == st.session_state.selected_age_cat]
    if st.session_state.selected_outcome is not None:
        filtered_df = filtered_df[filtered_df['Outcome'] == st.session_state.selected_outcome]
    
    st.write("---")
    st.write("**Preview Data Terfilter (5 Baris Teratas)**")
    st.dataframe(filtered_df.head(5))

    # 3. Pie Charts (Kategorik)
    col1, col2 = st.columns([5,5])
    def pie_chart_gen(data, col_name, title):
        counts = data[col_name].value_counts().reset_index()
        counts.columns = [col_name, 'count']
        return alt.Chart(counts).mark_arc(innerRadius=50).encode(
            theta='count:Q',
            color=alt.Color(f'{col_name}:N', scale=alt.Scale(scheme='tableau10')),
            tooltip=[col_name, 'count']
        ).properties(height=300, title=title)

    with col1:
        st.altair_chart(pie_chart_gen(filtered_df, 'bmi_category', 'Distribusi Kategori BMI'), use_container_width=True)
    with col2:
        st.altair_chart(pie_chart_gen(filtered_df, 'age_category', 'Distribusi Kategori Usia'), use_container_width=True)

    # 4. Histogram & Scatter Plots
    st.write("### Analisis Faktor Risiko")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribusi Kadar Glukosa**")
        glucose_hist = alt.Chart(filtered_df).mark_bar(color='teal').encode(
            alt.X('Glucose:Q', bin=alt.Bin(maxbins=20), title='Glukosa'),
            alt.Y('count():Q', title='Jumlah Pasien')
        ).properties(height=300)
        st.altair_chart(glucose_hist, use_container_width=True)

    with col2:
        st.write("**Hubungan Glukosa & BMI**")
        scatter = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X('Glucose:Q', title='Kadar Glukosa'),
            y=alt.Y('BMI:Q', title='BMI'),
            color=alt.Color('Outcome:N', title='Diabetes', scale=alt.Scale(range=['#1f77b4', '#d62728'])),
            tooltip=['Age', 'BMI', 'Glucose', 'Outcome']
        ).interactive().properties(height=300)
        st.altair_chart(scatter, use_container_width=True)

    # 5. Box Plot & Line Chart (Tren Usia)
    st.write("### Distribusi Penyakit Berdasarkan Usia")
    
    box = alt.Chart(filtered_df).mark_boxplot(extent=1.5).encode(
        x=alt.X('Outcome:N', title='Status Diabetes (0=Sehat, 1=Diabetes)'),
        y=alt.Y('Age:Q', title='Usia'),
        color='Outcome:N'
    ).properties(height=350)
    st.altair_chart(box, use_container_width=True)

    # Line Chart: Rata-rata Glukosa per Kelompok Usia
    st.write("**Tren Rata-rata Glukosa Berdasarkan Usia**")
    age_trend = filtered_df.groupby('Age')['Glucose'].mean().reset_index()
    line_age = alt.Chart(age_trend).mark_line(point=True, color='red').encode(
        x='Age:Q',
        y=alt.Y('Glucose:Q', title='Rata-rata Glukosa'),
        tooltip=['Age', 'Glucose']
    ).properties(height=350)
    st.altair_chart(line_age, use_container_width=True)

    # 6. Analisis Berdasarkan Kategori (Bar Charts)
    st.write("### Perbandingan Berdasarkan Kategori")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Diabetes Berdasarkan Kategori BMI**")
        bmi_outcome = filtered_df.groupby(['bmi_category', 'Outcome']).size().reset_index(name='counts')
        bmi_outcome['Outcome'] = bmi_outcome['Outcome'].map({0: 'Sehat', 1: 'Diabetes'})
        chart_bmi = alt.Chart(bmi_outcome).mark_bar().encode(
            x=alt.X('bmi_category:N', title='Kategori BMI'),
            y='counts:Q',
            color='Outcome:N',
            tooltip=['bmi_category', 'counts', 'Outcome']
        ).properties(height=300)
        st.altair_chart(chart_bmi, use_container_width=True)

    with col2:
        st.write("**Diabetes Berdasarkan Jumlah Kehamilan**")
        preg_outcome = filtered_df.groupby(['Pregnancies', 'Outcome']).size().reset_index(name='counts')
        preg_outcome['Outcome'] = preg_outcome['Outcome'].map({0: 'Sehat', 1: 'Diabetes'})
        chart_preg = alt.Chart(preg_outcome).mark_line(point=True).encode(
            x=alt.X('Pregnancies:O', title='Jumlah Kehamilan'),
            y='counts:Q',
            color='Outcome:N',
            tooltip=['Pregnancies', 'counts']
        ).properties(height=300)
        st.altair_chart(chart_preg, use_container_width=True)
def chart():
    chart_diabetes()

if __name__ == "__main__":
    chart_diabetes()
    
    