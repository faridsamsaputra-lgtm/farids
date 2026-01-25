import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def chart_diabetes():

    df = pd.read_csv('diabetes (2).csv')

    # ===============================
    # PREPROCESSING RINGAN
    # ===============================
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    df['bmi_category'] = pd.cut(
        df['BMI'], bins=[0,18.5,25,30,100],
        labels=['Underweight','Normal','Overweight','Obesity']
    )

    df['age_category'] = pd.cut(
        df['Age'], bins=[0,30,50,100],
        labels=['Young Adult','Middle Aged','Senior']
    )

    # ===============================
    # JUDUL
    # ===============================
    st.title("📊 Dashboard Monitoring Pasien Diabetes")

    total_pasien = df.shape[0]
    total_diabetes = df['Outcome'].sum()
    diabetes_rate = (total_diabetes / total_pasien) * 100

    # ===============================
    # METRIC
    # ===============================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pasien", total_pasien)
    col2.metric("Pasien Diabetes", total_diabetes)
    col3.metric("Prevalensi", f"{diabetes_rate:.2f}%")

    st.info(
        "📌 **Interpretasi:** "
        "Prevalensi menunjukkan proporsi pasien yang terindikasi diabetes "
        "terhadap keseluruhan populasi data."
    )

    st.divider()

    # ===============================
    # PIE CHART
    # ===============================
    st.subheader("Distribusi Karakteristik Pasien")

    col1, col2 = st.columns(2)

    def pie_chart(data, col, title):
        c = data[col].value_counts().reset_index()
        c.columns = [col, 'Jumlah']
        return alt.Chart(c).mark_arc(innerRadius=50).encode(
            theta='Jumlah:Q',
            color=col,
            tooltip=[col, 'Jumlah']
        ).properties(title=title, height=300)

    with col1:
        st.altair_chart(pie_chart(df, 'bmi_category', 'Kategori BMI'), use_container_width=True)
        st.success(
            "Interpretasi: Mayoritas pasien berada pada kategori "
            "Overweight dan Obesity yang merupakan faktor risiko utama diabetes."
        )

    with col2:
        st.altair_chart(pie_chart(df, 'age_category', 'Kategori Usia'), use_container_width=True)
        st.success(
            "Interpretasi: Risiko diabetes cenderung meningkat pada kelompok usia menengah hingga lanjut."
        )

    st.divider()

    # ===============================
    # HISTOGRAM & SCATTER
    # ===============================
    st.subheader("Analisis Faktor Risiko")

    col1, col2 = st.columns(2)

    with col1:
        hist = alt.Chart(df).mark_bar().encode(
            alt.X('Glucose:Q', bin=alt.Bin(maxbins=25)),
            y='count()'
        ).properties(height=300)
        st.altair_chart(hist, use_container_width=True)

        st.info(
            "Interpretasi: Distribusi glukosa menunjukkan konsentrasi tinggi "
            "pada nilai di atas normal yang mengindikasikan risiko diabetes."
        )

    with col2:
        scatter = alt.Chart(df).mark_circle(size=60).encode(
            x='Glucose:Q',
            y='BMI:Q',
            color='Outcome:N',
            tooltip=['Age','BMI','Glucose','Outcome']
        ).interactive().properties(height=300)

        st.altair_chart(scatter, use_container_width=True)

        st.info(
            "Interpretasi: Pasien dengan kadar glukosa dan BMI tinggi "
            "lebih banyak berada pada kelompok diabetes."
        )

    st.divider()

    # ===============================
    # BOXPLOT
    # ===============================
    st.subheader("Distribusi Usia terhadap Status Diabetes")

    box = alt.Chart(df).mark_boxplot().encode(
        x='Outcome:N',
        y='Age:Q',
        color='Outcome:N'
    ).properties(height=350)

    st.altair_chart(box, use_container_width=True)

    st.warning(
        "Interpretasi: Median usia pasien diabetes lebih tinggi "
        "dibandingkan pasien non-diabetes."
    )

    # ===============================
    # LINE CHART
    # ===============================
    st.subheader("Tren Rata-rata Glukosa Berdasarkan Usia")

    trend = df.groupby('Age')['Glucose'].mean().reset_index()

    line = alt.Chart(trend).mark_line(point=True).encode(
        x='Age:Q',
        y='Glucose:Q'
    ).properties(height=350)

    st.altair_chart(line, use_container_width=True)

    st.info(
        "Interpretasi: Terlihat tren peningkatan kadar glukosa "
        "seiring bertambahnya usia."
    )

    st.divider()

    # ===============================
    # BAR CATEGORY
    # ===============================
    st.subheader("Perbandingan Berdasarkan Kategori")

    col1, col2 = st.columns(2)

    with col1:
        bmi_out = df.groupby(['bmi_category','Outcome']).size().reset_index(name='Jumlah')
        bmi_out['Outcome'] = bmi_out['Outcome'].map({0:'Sehat',1:'Diabetes'})

        chart_bmi = alt.Chart(bmi_out).mark_bar().encode(
            x='bmi_category:N',
            y='Jumlah:Q',
            color='Outcome:N'
        ).properties(height=300)

        st.altair_chart(chart_bmi, use_container_width=True)

        st.success(
            "Interpretasi: Kasus diabetes paling banyak ditemukan "
            "pada kategori Obesity."
        )

    with col2:
        preg = df.groupby(['Pregnancies','Outcome']).size().reset_index(name='Jumlah')
        preg['Outcome'] = preg['Outcome'].map({0:'Sehat',1:'Diabetes'})

        chart_preg = alt.Chart(preg).mark_line(point=True).encode(
            x='Pregnancies:O',
            y='Jumlah:Q',
            color='Outcome:N'
        ).properties(height=300)

        st.altair_chart(chart_preg, use_container_width=True)

        st.success(
            "Interpretasi: Semakin banyak jumlah kehamilan, "
            "risiko diabetes cenderung meningkat."
        )


def chart():
    chart_diabetes()


if __name__ == "__main__":
    chart_diabetes()
