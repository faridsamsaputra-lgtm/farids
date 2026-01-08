import streamlit as st



def about_dataset():

    st.write('### **Tentang Dataset Diabetes**')

    col1, col2 = st.columns([10, 10])



    with col1:

        # Menggunakan gambar yang relevan dengan pengecekan glukosa/diabetes

        link = "farid.png"

        # Menambahkan parameter 'width=300' untuk memperkecil ukuran gambar

        st.image(link, caption="Ilustrasi Pemeriksaan Glukosa Darah", use_container_width=False, width=200)



    with col2:

        st.write("""

        Menurut **International Diabetes Federation (IDF)**, diabetes adalah salah satu keadaan 

        darurat kesehatan global dengan pertumbuhan tercepat di abad ke-21. Jika tidak dikelola, 

        diabetes dapat menyebabkan komplikasi serius seperti penyakit jantung, gagal ginjal, dan kebutaan.



        Dataset ini aslinya berasal dari **National Institute of Diabetes and Digestive and Kidney Diseases**. 

        Tujuannya adalah untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes 

        berdasarkan pengukuran klinis tertentu yang dikumpulkan dari pasien wanita keturunan India Pima.

        """)



    st.write("---")

    st.write("**Informasi Atribut/Fitur:**")

    

    # Menjelaskan kolom-kolom yang ada di diabetes (2).csv

    col_desc1, col_desc2 = st.columns(2)

    

    with col_desc1:

        st.markdown("""

        * **Pregnancies**: Jumlah berapa kali hamil.

        * **Glucose**: Konsentrasi glukosa plasma dalam tes toleransi glukosa oral.

        * **Blood Pressure**: Tekanan darah diastolik (mm Hg).

        * **Skin Thickness**: Ketebalan lipatan kulit trisep (mm).

        """)

        

    with col_desc2:

        st.markdown("""

        * **Insulin**: Kadar insulin serum 2 jam (mu U/ml).

        * **BMI**: Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2).

        * **Diabetes Pedigree Function**: Fungsi yang menunjukkan skor riwayat genetik diabetes.

        * **Age**: Usia pasien (tahun).

        * **Outcome**: Variabel target (0: Tidak Diabetes, 1: Diabetes).

        """)



    st.write("---")

    st.write("**Langkah-langkah Metode Logistic Regression:**")

    st.markdown("""

    Dalam memprediksi risiko diabetes, sistem mengikuti tahapan berikut:

    1. **Data Cleaning**: Menangani nilai nol (0) pada kolom medis yang tidak logis (seperti BMI atau Insulin) dengan nilai median.

    2. **Feature Selection**: Mengambil 8 parameter medis utama sebagai input prediktor.

    3. **Data Scaling**: Menyamakan skala data menggunakan *MinMaxScaler* agar angka yang besar (seperti Insulin) tidak mendominasi angka kecil (seperti DPF).

    4. **Model Training**: Menggunakan fungsi **Sigmoid** untuk memetakan input data ke dalam probabilitas antara 0 dan 1.

    5. **Klasifikasi**: Menentukan ambang batas (*threshold*) 0.5. Jika probabilitas $\geq 0.5$, maka diklasifikasikan sebagai **Diabetes**.

    """)



    st.info("Metode analisis yang digunakan dalam aplikasi ini adalah **Logistic Regression** untuk klasifikasi risiko biner.")