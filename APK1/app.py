import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Risiko Diabetes", page_icon="🩺", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('Diabetes.csv')
    return df

@st.cache_resource
def train_model(df):
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    # Train-test split (80:20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Pipeline dengan StandardScaler dan RandomForestClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test

# Load Data dan Train Model
try:
    data = load_data()
    model_pipeline, X_test, y_test = train_model(data)
    data_loaded = True
except Exception as e:
    st.error(f"Gagal memuat data atau melatih model: {e}")
    data_loaded = False

if data_loaded:
    # Sidebar untuk Navigasi
    st.sidebar.title("🩺 Navigasi")
    menu = st.sidebar.radio("Pilih Menu:", ["Home", "Dataset", "Evaluasi Model", "Eksperimen Model", "Prediksi"])

    if menu == "Home":
        st.title("Aplikasi Prediksi Risiko Penyakit Diabetes")
        st.write("### Selamat datang di Aplikasi Prediksi Risiko Diabetes!")
        st.write("""
        Website ini merupakan sebuah sistem cerdas berbasis *Machine Learning* yang dirancang untuk melakukan skrining dan prediksi dini terhadap risiko penyakit Diabetes. 
        Dengan memasukkan data metrik kesehatan Anda, kecerdasan buatan akan membandingkan kondisi Anda dengan pola ribuan data historis pasien lainnya.
        """)

        st.subheader("Bagaimana Cara Kerja Sistem Ini?")
        st.markdown("""
        1. **Pengumpulan & Pemrosesan Data**: Sistem dilatih menggunakan dataset yang berisi berbagai atribut vital (seperti umur, BMI, kadar HbA1c, glukosa, dll). Data ini distandarisasi (*Standard Scaling*) agar model tidak bias terhadap angka yang terlalu besar atau kecil.
        2. **Analisis Model (Random Forest)**: Data Anda akan diproses melalui algoritma *Random Forest*, yaitu metode *Machine Learning* yang membangun banyak 'pohon keputusan' secara bersamaan, kemudian melakukan _voting_ untuk menghasilkan prediksi akhir yang akurat dan kuat.
        3. **Hasil Prediksi Otomatis**: Dalam hitungan detik, algoritma dapat mengklasifikasikan apakah metrik kesehatan Anda memiliki kemiripan dengan pola pasien penderita Diabetes atau tidak.
        """)
        
        st.subheader("Penjelasan Navigasi")
        st.markdown("""
        - 🏠 **Home**: Beranda informasi tentang tujuan dan mekanisme kerja aplikasi.
        - 📊 **Dataset**: Halaman untuk melihat langsung referensi data primer yang digunakan, termasuk dimensi dan statistik data.
        - 📈 **Evaluasi Model**: Halaman teknis yang menampilkan tingkat keakuratan sistem (*Accuracy*), laporan klasifikasi, dan *Confusion Matrix* dalam memisahkan kelas Positif dan Negatif.
        - 🩺 **Prediksi**: Fitur utama! Anda dapat bereksperimen dengan memasukkan profil kesehatan hipotetis atau kondisi asli Anda untuk menerima analisis risiko seketika.
        """)

    elif menu == "Dataset":
        st.title("Tinjauan Dataset")
        st.write("Dataset di bawah ini digunakan untuk melatih model Machine Learning.")
        
        st.subheader("Raw Data (20 Baris Pertama)")
        st.dataframe(data.head(20))
        
        st.subheader("Dimensi Data (Shape)")
        st.write(f"Dataset memiliki **{data.shape[0]}** baris dan **{data.shape[1]}** kolom.")
        
        st.subheader("Statistik Deskriptif")
        st.dataframe(data.describe())

    elif menu == "Evaluasi Model":
        st.title("Evaluasi Performa Model")
        st.write("Berikut adalah hasil evaluasi dari model Random Forest Classifier pada data uji (20%).")
        
        # Prediksi pada test set
        y_pred = model_pipeline.predict(X_test)
        
        # Akurasi
        acc = accuracy_score(y_test, y_pred)
        st.metric(label="Akurasi Model", value=f"{acc * 100:.2f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Report")
            # Konversi report ke dataframe untuk tampilan rapi
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.format("{:.3f}"))
            
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
            ax.set_ylabel('Aktual')
            ax.set_xlabel('Prediksi')
            st.pyplot(fig)

    elif menu == "Eksperimen Model":
        st.title("🔬 Eksperimen Hyperparameter & Preprocessing")
        st.write("Jalankan berbagai kombinasi preprocessing dan hyperparameter untuk membandingkan performa akhir model secara langsung.")
        
        # Pilihan preprocessing
        norm_options = st.multiselect(
            "Pilih Metode Normalisasi (bisa lebih dari satu):", 
            ["Tanpa Normalisasi", "Z-Score (StandardScaler)", "Min-Max Scaler"], 
            default=["Z-Score (StandardScaler)", "Tanpa Normalisasi"]
        )
        
        # Pilihan n_estimators
        n_est_options = st.multiselect(
            "Pilih Jumlah Pohon (n_estimators) Random Forest:", 
            [50, 100, 200], 
            default=[50, 100]
        )
        
        if st.button("Jalankan Eksperimen", type="primary"):
            if not norm_options or not n_est_options:
                st.warning("Pilih minimal satu metode normalisasi dan satu nilai n_estimators.")
            else:
                with st.spinner("Sedang melatih dan mengevaluasi model... Ini mungkin membutuhkan waktu beberapa saat."):
                    X = data.drop('diabetes', axis=1)
                    y = data['diabetes']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                    
                    results = []
                    
                    for norm in norm_options:
                        for n_est in n_est_options:
                            steps = []
                            if norm == "Z-Score (StandardScaler)":
                                steps.append(('scaler', StandardScaler()))
                            elif norm == "Min-Max Scaler":
                                steps.append(('scaler', MinMaxScaler()))
                            
                            steps.append(('rf', RandomForestClassifier(n_estimators=n_est, random_state=42)))
                            
                            exp_pipeline = Pipeline(steps)
                            exp_pipeline.fit(X_train, y_train)
                            y_pred_exp = exp_pipeline.predict(X_test)
                            acc = accuracy_score(y_test, y_pred_exp)
                            
                            results.append({
                                "Metode Normalisasi": norm,
                                "n_estimators": n_est,
                                "Akurasi (%)": round(acc * 100, 4)
                            })
                    
                    df_results = pd.DataFrame(results)
                    st.success("✅ Eksperimen selesai!")
                    
                    st.markdown("---")
                    st.subheader("📊 Tabel Perbandingan Hasil")
                    st.dataframe(df_results.sort_values(by="Akurasi (%)", ascending=False).style.highlight_max(subset=['Akurasi (%)'], color='lightgreen'))
                    
                    st.subheader("📈 Visualisasi Perbandingan")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Gunakan barplot untuk membandingkan secara jelas
                    sns.barplot(data=df_results, x="Metode Normalisasi", y="Akurasi (%)", hue="n_estimators", ax=ax, palette="viridis")
                    
                    # Dinamis Y-axis agar perbedaan kecil bisa terlihat lebih jelas
                    min_acc = df_results['Akurasi (%)'].min()
                    max_acc = df_results['Akurasi (%)'].max()
                    ax.set_ylim(max(0, min_acc - 0.5), min(100, max_acc + 0.5))
                    
                    plt.title("Perbandingan Akurasi Model Berdasarkan Normalisasi dan n_estimators")
                    plt.legend(title='n_estimators')
                    st.pyplot(fig)

    elif menu == "Prediksi":
        st.title("Prediksi Interaktif Risiko Diabetes")
        st.write("Masukkan nilai fitur-fitur kesehatan pada form di bawah ini untuk melihat hasil prediksi.")
        
        # Ambil nilai unik dari dataset untuk opsi dropdown jika memungkinan
        gender_options = sorted(data['gender'].unique().tolist())
        hypertension_options = sorted(data['hypertension'].unique().tolist())
        heart_disease_options = sorted(data['heart_disease'].unique().tolist())
        smoking_options = sorted(data['smoking_history'].unique().tolist())
        
        # Kamus mapping untuk memperjelas pilihan dropdown
        format_gender = lambda x: {0: "Wanita", 1: "Pria"}.get(x, f"Lainnya ({x})")
        format_bool = lambda x: {0: "Tidak Ada (0)", 1: "Ada (1)"}.get(x, f"Kategori {x}")
        format_smoking = lambda x: f"Kategori {x}" # Tetap tampilkan kategori aslinya untuk smoking history

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Gender (Jenis Kelamin)", options=gender_options, format_func=format_gender, help="0 = Wanita, 1 = Pria")
                age = st.number_input("Age (Umur)", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
                hypertension = st.selectbox("Hypertension (Darah Tinggi)", options=hypertension_options, format_func=format_bool, help="0 = Tidak ada riwayat, 1 = Ada riwayat")
                heart_disease = st.selectbox("Heart Disease (Penyakit Jantung)", options=heart_disease_options, format_func=format_bool, help="0 = Tidak ada riwayat, 1 = Ada riwayat")
                
            with col2:
                smoking_history = st.selectbox("Smoking History (Riwayat Merokok)", options=smoking_options, format_func=format_smoking, help="Tingkatan kategori riwayat merokok berdasarkan dataset (0-5)")
                bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
                hba1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
                blood_glucose_level = st.number_input("Blood Glucose Level (Tingkat Glukosa)", min_value=0, max_value=500, value=100, step=1)
                
            submit_button = st.form_submit_button(label="Lakukan Prediksi", type="primary")
            
        if submit_button:
            # Susun array input
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'smoking_history': [smoking_history],
                'bmi': [bmi],
                'HbA1c_level': [hba1c_level],
                'blood_glucose_level': [blood_glucose_level]
            })
            
            # Pastikan urutan kolom sesuai dengan saat training
            expected_cols = model_pipeline.named_steps['rf'].feature_names_in_ if hasattr(model_pipeline.named_steps['rf'], 'feature_names_in_') else model_pipeline.feature_names_in_
            
            # Predict
            prediction = model_pipeline.predict(input_data)[0]
            
            st.markdown("---")
            st.subheader("Hasil Prediksi:")
            
            if prediction == 0:
                st.success("Tanda vital menunjukkan hasil **NEGATIF** untuk Diabetes. Tetap jaga pola makan dan gaya hidup sehat!")
            else:
                st.error("Tanda vital menunjukkan hasil **POSITIF** atau risiko tinggi untuk Diabetes. Disarankan untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut.")
