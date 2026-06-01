import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, RobustScaler
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Machine Learning Explorer", layout="wide", page_icon="📈")

# -----------------
# 1. HELPER FUNCS
import os

@st.cache_data
def load_data(dataset_name):
    try:
        # Mengambil lokasi direktori dari app.py ini secara absolut
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if dataset_name == "Diabetes":
            file_path = os.path.join(base_dir, "Diabetes.csv")
            df = pd.read_csv(file_path)
            target = "diabetes"
        else:
            file_path = os.path.join(base_dir, "obesitas.csv")
            df = pd.read_csv(file_path)
            target = "NObeyesdad"
        return df, target
    except Exception as e:
        st.error(f"Gagal memuat dataset {dataset_name}. Pastikan file berada di direktori yang sama. Error: {e}")
        return pd.DataFrame(), ""

# -----------------
# 2. SIDEBAR
# -----------------
st.sidebar.title("Konfigurasi Eksperimen")

dataset_name = st.sidebar.selectbox("Pilih Dataset", ["Diabetes", "Obesitas"])
algo = st.sidebar.selectbox("Pilih Algoritma", [
    "Logistic Regression", 
    "K-Nearest Neighbors (KNN)", 
    "Random Forest", 
    "AdaBoost"
])

norm_options = ["Tanpa Normalisasi", "MinMax", "Z-Score", "Norm L1", "Norm L2"]
if algo == "Logistic Regression":
    norm_options.extend(["MaxAbs", "Robust Scaler"])
norm_choice = st.sidebar.selectbox("Pilih Normalisasi", norm_options)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameter Tuning")

params = {}
model_invalid = False
invalid_reason = ""

if algo == "Logistic Regression":
    solver = st.sidebar.selectbox("solver", ["lbfgs", "liblinear", "saga", "sag", "newton-cg"])
    
    # Validation logic
    valid_penalties = []
    if solver in ["lbfgs", "sag", "newton-cg"]:
        valid_penalties = ["l2", "none"]
    elif solver == "liblinear":
        valid_penalties = ["l1", "l2"]
    elif solver == "saga":
        valid_penalties = ["l1", "l2", "elasticnet", "none"]
        
    penalty = st.sidebar.selectbox("penalty", valid_penalties)
    c_val = st.sidebar.selectbox("C", [0.1, 1.0, 10.0])
    multi_class = st.sidebar.selectbox("multi_class", ["ovr", "multinomial"])
    
    if solver == "liblinear" and multi_class == "multinomial":
        st.sidebar.warning("Solver 'liblinear' tidak mendukung multi_class='multinomial'. Menggunakan 'ovr' secara otomatis.")
        multi_class = "ovr"
        
    max_iter = st.sidebar.selectbox("max_iter", [100, 500])
    
    # Setup params
    params = {
        'solver': solver,
        'penalty': None if penalty == "none" else penalty,
        'C': c_val,
        # 'multi_class': multi_class, # Dihapus dari parameter karena Scikit-Learn 1.8.0 tidak mendukung kwargs ini
        'max_iter': max_iter
    }

    if penalty == "elasticnet":
        l1_ratio = st.sidebar.slider("l1_ratio", 0.0, 1.0, 0.5)
        params['l1_ratio'] = l1_ratio

elif algo == "K-Nearest Neighbors (KNN)":
    k = st.sidebar.slider("n_neighbors (K)", 1, 10, 5)
    metric = st.sidebar.selectbox("metric", ["Euclidean", "Manhattan"])
    params = {
        'n_neighbors': k, 
        'metric': 'euclidean' if metric == "Euclidean" else 'manhattan'
    }

elif algo == "Random Forest":
    n_est = st.sidebar.selectbox("n_estimators", [100, 200, 300])
    max_d = st.sidebar.selectbox("max_depth", [10, 15, 20])
    min_split = st.sidebar.selectbox("min_samples_split", [2, 5, 10])
    min_leaf = st.sidebar.selectbox("min_samples_leaf", [1, 2, 4])
    max_feat = st.sidebar.selectbox("max_features", ["sqrt", "log2"])
    params = {
        'n_estimators': n_est,
        'max_depth': max_d,
        'min_samples_split': min_split,
        'min_samples_leaf': min_leaf,
        'max_features': max_feat,
        'random_state': 42
    }

elif algo == "AdaBoost":
    n_est = st.sidebar.selectbox("n_estimators", [50, 100, 150])
    lr = st.sidebar.selectbox("learning_rate", [0.1, 0.5, 1.0])
    params = {
        'n_estimators': n_est, 
        'learning_rate': lr,
        'random_state': 42
    }

# -----------------
# 3. PREP DATA
# -----------------
df, target_col = load_data(dataset_name)

if df.empty:
    st.stop()

# Build pipeline components
scaler = None
if norm_choice == "MinMax":
    scaler = MinMaxScaler()
elif norm_choice == "Z-Score":
    scaler = StandardScaler()
elif norm_choice == "Norm L1":
    scaler = Normalizer(norm='l1')
elif norm_choice == "Norm L2":
    scaler = Normalizer(norm='l2')
elif norm_choice == "MaxAbs":
    scaler = MaxAbsScaler()
elif norm_choice == "Robust Scaler":
    scaler = RobustScaler()

if algo == "Logistic Regression":
    model = LogisticRegression(**params)
elif algo == "K-Nearest Neighbors (KNN)":
    model = KNeighborsClassifier(**params)
elif algo == "Random Forest":
    model = RandomForestClassifier(**params)
elif algo == "AdaBoost":
    model = AdaBoostClassifier(**params)

steps = []
if scaler is not None:
    steps.append(('scaler', scaler))
steps.append(('model', model))
pipeline = Pipeline(steps)

# Split data
X = df.drop(columns=[target_col])
y = df[target_col]

# Check target cardinality for LogisticRegression ovr vs multinomial issues if any
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
except ValueError as e:
    # Fallback if stratify fails due to too few samples in a class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train model globally to be used in Evaluasi and Prediksi
model_trained = False
error_msg = ""
try:
    pipeline.fit(X_train, y_train)
    model_trained = True
except Exception as e:
    error_msg = str(e)


# -----------------
# 4. MAIN PAGE TABS
# -----------------
st.title(f"Analisis ML: Dataset {dataset_name}")
st.markdown("Selamat datang di platform interaktif untuk mengeksplorasi dan mengevaluasi bagaimana model pembelajaran mesin (Machine Learning) mengolah profil data kesehatan.")

tab1, tab2, tab3 = st.tabs(["Edukasi & Konsep", "Evaluasi & Grafik Visualisasi", "Simulasi Prediksi"])

with tab1:
    st.header("Metode Penyesuaian Data (Normalisasi)")
    st.markdown("Normalisasi adalah proses menyamakan skala rentang angka dalam dataset agar model Machine Learning dapat belajar dengan lebih optimal dan tidak bias terhadap nilai-nilai yang jauh berbeda (contoh: berat badan versus umur).")
    if norm_choice == "Tanpa Normalisasi":
        st.info("💡 **Tanpa Normalisasi**: Mempertahankan nilai asli pada dataset secara utuh tanpa penyesuaian perhitungan.")
    elif norm_choice == "MinMax":
        st.info("💡 **MinMax Scaler**: Memampatkan rentang semua nilai menjadi skala proporsional antara 0 hingga 1. Metode ini mempertahankan sebaran distribusi asli namun rentan terhadap nilai ekstrem (*outlier*).")
    elif norm_choice == "Z-Score":
        st.info("💡 **Z-Score (Standarisasi)**: Mengubah data dengan menyesuaikan nilai rata-rata menjadi 0 dan standar deviasi menjadi 1. Sangat disarankan untuk mengatasi rentang variasi yang terlalu luas antar variabel.")
    elif norm_choice == "Norm L1":
        st.info("💡 **Norm L1**: Memastikan total rasio nilai absolut dalam satu baris sampel data disetarakan menjadi persentase bulat bernilai 1.")
    elif norm_choice == "Norm L2":
        st.info("💡 **Norm L2**: Memastikan total akar kuadrat nilai dalam satu baris data menjadi 1. Sering digunakan jika target pengamatan berfokus pada penghitungan tingkat kemiripan metrik.")
    elif norm_choice == "MaxAbs":
        st.info("💡 **MaxAbs Scaler**: Membagi semua nilai dalam matriks data dengan nilai absolut tertingginya sehingga angka maksimum adalah 1 atau -1. Berguna jika mayoritas isi dataset bernilai nol (sparse data).")
    elif norm_choice == "Robust Scaler":
        st.info("💡 **Robust Scaler**: Menggunakan rumusan matematis kebal anomali (seperti metrik kuartil/median). Sangat direkomendasikan apabila dataset Anda dinilai rawan memiliki kasus-kasus ekstrem/pencilan (*outlier*).")

    st.header("Cara Kerja Model Prediksi (Algoritma)")
    if algo == "Logistic Regression":
        st.success("🤖 **Logistic Regression**: Algoritma statistik yang mendasarkan perhitungannya pada probabilitas matematis (peluang kejadian). Model akan mengevaluasi bobot dari masing-masing indikator dan mengklasifikasikan apakah suatu data condong pada hasil Positif atau Negatif.")
    elif algo == "K-Nearest Neighbors (KNN)":
        st.success("🤖 **K-Nearest Neighbors (KNN)**: Algoritma klasifikasi prediktif yang bekerja dengan prinsip kedekatan identitas. Model akan melihat sekumpulan kasus sebelumnya (tetangga) dengan indikator terdekat dan mengambil klasifikasi yang paling dominan di antara mereka.")
    elif algo == "Random Forest":
        st.success("🤖 **Random Forest**: Pendekatan algoritma kolektif (*ensemble learning*) di mana mesin membentuk ratusan Pohon Keputusan (*Decision Trees*) secara acak, kemudian menentukan hasil klasifikasi final dengan sistem jajak pendapat mayoritas (*voting*) guna memastikan performa akurasi yang lebih andal.")
    elif algo == "AdaBoost":
        st.success("🤖 **AdaBoost**: Teknik *ensemble learning* yang beroperasi secara iteratif. Mesin menyusun banyak sistem prediktor sederhana, dan setiap estimator yang tercipta pada tahapan berikutnya ditugaskan untuk memperbaiki celah kesalahan pengklasifikasian yang dihasilkan dari estimator sebelumnya.")

with tab2:
    st.header("Evaluasi Kinerja Model")
    st.markdown("Pada bagian ini, model akan diuji menggunakan porsi dataset yang sengaja disisihkan dari awal (*testing set*) untuk mengukur tingkat ketepatan klasifikasi secara objektif.")
    if not model_trained:
        st.error(f"Terjadi kegagalan dalam proses pelatihan model. Pesan Kesalahan: {error_msg}")
        st.warning("Hal ini umumnya disebabkan oleh kombinasi parameter algoritma yang belum sesuai dengan distribusi data yang dianalisis. Silakan ubah opsi konfigurasi di panel kiri.")
    else:
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Metric Akurasi
        st.metric(label="Tingkat Akurasi Total (Rasio kebenaran prediksi dari seluruh sampel yang diuji)", value=f"{acc * 100:.2f} %")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Report")
            st.markdown("""
            * **Precision**: Seberapa akurat model ketika memberikan label diagnosis tertentu? (Minimisasi kesalahan '*False Positive*')
            * **Recall**: Seberapa sensitif model mendeteksi keseluruhan sampel positif secara faktual? (Minimisasi kesalahan '*False Negative*')
            * **F1-Score**: Rata-rata yang menyeimbangkan metrik nilai Precision dan metrik Recall.
            """)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            
        with col2:
            st.subheader("Confusion Matrix")
            st.markdown("Matriks yang memvisualisasikan jumlah keselarasan klasifikasi (ditandai pada garis sel diagonal utama) dibandingkan jumlah kelas yang salah diprediksi.")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
            ax_cm.set_xlabel('Prediksi Model')
            ax_cm.set_ylabel('Kenyataan Faktual')
            st.pyplot(fig_cm)
            
        st.markdown("---")
        st.subheader("Perbandingan Precision, Recall, & F1-Score per Kelas")
        
        # Prepare data for Bar Chart
        classes = [c for c in report_dict.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics_data = {
            'Kelas': [],
            'Metric': [],
            'Score': []
        }
        for cls in classes:
            metrics_data['Kelas'].extend([str(cls), str(cls), str(cls)])
            metrics_data['Metric'].extend(['Precision', 'Recall', 'F1-Score'])
            metrics_data['Score'].extend([
                report_dict[cls]['precision'],
                report_dict[cls]['recall'],
                report_dict[cls]['f1-score']
            ])
            
        df_metrics = pd.DataFrame(metrics_data)
        
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_metrics, x='Kelas', y='Score', hue='Metric', ax=ax_bar, palette='viridis')
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_title("Metrics per Kelas", fontsize=14)
        ax_bar.set_ylabel("Score", fontsize=12)
        ax_bar.set_xlabel("Kelas Target", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig_bar)

with tab3:
    st.header("Simulasi Prediksi Kustom")
    st.markdown("Silakan sesuaikan konfigurasi nilai indikator (fitur) di bawah ini. Model pembelajaran mesin yang telah dievaluasi tadi akan menganalisis keseluruhan masukan data Anda secara aktual (*real-time*) untuk memberikan hasil prediksi medis.")
    
    if not model_trained:
        st.error("Model belum dapat dioperasikan. Silakan periksa kembali konfigurasi parameter di panel kiri.")
    else:
        with st.form("prediction_form"):
            features = X.columns
            # Determine UI columns (e.g., 3 columns)
            cols = st.columns(3)
            
            def format_label(feat, val):
                feat_lower = feat.lower()
                if feat_lower == 'gender':
                    return f"{val} (Perempuan)" if val == 0 else f"{val} (Laki-laki)" if val == 1 else str(val)
                elif feat_lower in ['hypertension', 'heart_disease', 'family_history_with_overweight', 'favc', 'smoke', 'scc']:
                    return f"{val} (Tidak)" if val == 0 else f"{val} (Ya)" if val == 1 else str(val)
                return str(val)

            input_data = {}
            for i, feat in enumerate(features):
                # Identify if feature is likely categorical (small number of unique values) or continuous
                unique_vals = sorted(X[feat].dropna().unique())
                col_idx = i % 3
                with cols[col_idx]:
                    if len(unique_vals) <= 10:
                        # Categorical or light discrete -> selectbox
                        input_data[feat] = st.selectbox(feat, options=unique_vals, format_func=lambda x, f=feat: format_label(f, x))
                    else:
                        # Continuous numeric -> number_input
                        min_val = float(X[feat].min())
                        max_val = float(X[feat].max())
                        mean_val = float(X[feat].mean())
                        input_data[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=mean_val)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Jalankan Prediksi Analisis")
            
            if submitted:
                # Convert input dictionary to DataFrame
                input_df = pd.DataFrame([input_data])
                try:
                    prediction = pipeline.predict(input_df)[0]
                    
                    st.markdown("### Hasil Analisis Prediksi:")
                    if dataset_name == "Diabetes":
                        if prediction == 0:
                            st.success("Berdasarkan proses evaluasi atas susunan indikator tersebut, klasifikasi profil diprediksi sebagai: **Kategori Normal (Tidak Berisiko Tinggi Diabetes)**.")
                        else:
                            st.error("Berdasarkan evaluasi indikator medis tersebut, profil divalidasi memiliki klasifikasi **Berisiko Tinggi Diabetes**. Saran klinis profesional sangat direkomendasikan.")
                    else:
                        # Obesitas
                        if prediction == 0 or str(prediction).lower() == 'normal':
                            st.success(f"Profil data ini diklasifikasikan pada tingkat berat badan: **Normal** (Label Metrik: {prediction}).")
                        else:
                            st.warning(f"Profil data ini diklasifikasikan termasuk ke dalam skala indikasi obesitas pada tingkat (kategori klasifikasi): **{prediction}**. Metrik skala obesitas yang bertambah merepresentasikan level penumpukan massa tubuh yang lebih tinggi secara proporsional.")
                except Exception as e:
                    st.error(f"Sistem gagal mengeksekusi operasi evaluasi prediksi. Rincian masalah teknis: {e}")
