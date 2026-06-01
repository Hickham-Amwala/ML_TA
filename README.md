# 🚀 ML Explorer: Prediksi & Analisis Risiko Kesehatan Terpadu

Sebuah aplikasi web interaktif berbasis Streamlit yang dirancang secara khusus untuk mengeksplorasi, melatih, dan mengevaluasi berbagai algoritma *Machine Learning* (Pembelajaran Mesin) secara *real-time*.

Aplikasi ini menargetkan dua dataset medis utama:
1. **Risiko Diabetes**: Memprediksi apakah seseorang memiliki risiko tinggi terkena diabetes berdasarkan metrik kesehatan.
2. **Risiko Obesitas**: Mengklasifikasikan tingkat/level obesitas seseorang berdasarkan pola makan, fisik, dan gaya hidup.

---

## 🌟 Fitur Utama Aplikasi

1. **Dashboard Edukasi & Konsep (Tab 1)**
   - Menjabarkan dengan bahasa yang mudah dipahami tentang apa itu algoritma prediktif (seperti *Logistic Regression, KNN, Random Forest*, dan *AdaBoost*).
   - Menjelaskan fungsi *Normalisasi Data* (seperti *MinMax, Z-Score*, dsb) sebelum mesin memproses data, dianalogikan seperti menyamakan mata uang sebelum membandingkan harga.

2. **Evaluasi & Visualisasi Kinerja (Tab 2)**
   - **Hyperparameter Tuning**: Pengguna dapat mengubah parameter algoritma dan normalisasi di *sidebar* untuk melihat bagaimana model beradaptasi.
   - **Classification Report**: Menampilkan *raport kepintaran mesin* (Tingkat Akurasi, Precision, Recall, dan F1-Score).
   - **Confusion Matrix**: Matriks warna interaktif (*Heatmap*) yang membedah jumlah persis data tebakan yang benar (*True Positives/Negatives*) melawan tebakan yang meleset.

3. **Simulasi Prediksi Interaktif (Tab 3)**
   - Pengguna/Praktisi bisa bertindak seolah sedang mendiagnosis pasien. Cukup sesuaikan angka (fitur kondisi tubuh) melalui *form* input yang canggih dan praktis.
   - Hasil akan muncul secara instan disertai simpulan medis (Misal: *"Normal"* atau *"Berisiko Tinggi"*).
   - *User-Friendly*: Angka-angka biner kaku seperti `0` dan `1` pada pilihan *dropdown* telah diformat otomatis menjadi teks yang bisa dibaca manusia, misal `0 (Perempuan)` atau `1 (Ya)`.

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman**: Python 3
* **Antarmuka (Frontend/UI)**: Streamlit
* **Inti Pemrosesan Mesin (ML)**: Scikit-Learn
* **Visualisasi & Analisis Data**: Pandas, NumPy, Matplotlib, Seaborn

## 📂 Struktur Direktori Utama
Aplikasi ini dikonfigurasi untuk menangani pembacaan letak file secara cerdas dan otomatis.
* `APK2/app.py` - Berkas inti sistem aplikasi web *Streamlit*.
* `APK2/Diabetes.csv` - Kumpulan dataset indikator diabetes (Variabel target: `diabetes`).
* `APK2/obesitas.csv` - Kumpulan dataset tingkat/kategori kegemukan (Variabel target: `NObeyesdad`).

## 💻 Cara Menjalankan Secara Lokal (Localhost)

1. Pastikan Anda telah menginstal pustaka yang dibutuhkan:
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
   ```
2. Arahkan *terminal* (CMD/PowerShell) ke direktori Anda.
3. Jalankan perintah Streamlit:
   ```bash
   streamlit run APK2/app.py
   ```
   *(Catatan: Jika `streamlit` tidak terbaca di env sistem, gunakan `python -m streamlit run APK2/app.py`)*

---
*Proyek ini dikembangkan dalam rangka Tugas Akhir/Eksplorasi Pembelajaran Mesin untuk membedah kinerja berbagai model klasifikasi statistik.*
