# Analisis Sentimen Sederhana

Proyek ini adalah implementasi analisis sentimen sederhana menggunakan Python. Tujuannya untuk memahami dasar-dasar analisis sentimen pada data teks melalui tahapan pra‑pemrosesan, ekstraksi fitur, hingga klasifikasi.

# 🛍️ Analisis Sentimen Ulasan Marketplace Indonesia

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

Proyek ini adalah aplikasi web berbasis **Deep Learning** untuk menganalisis sentimen (Positif/Negatif) dari ulasan produk e-commerce di Indonesia. Model dilatih menggunakan dataset ulasan nyata dan mampu memahami konteks kalimat, termasuk bahasa gaul (slang) dan singkatan.

## 🎯 Fitur Utama
- **Real-time Analysis:** Input teks ulasan dan dapatkan hasil prediksi instan.
- **Deep Learning Model:** Menggunakan arsitektur Neural Network (Embedding + GlobalAveragePooling1D) yang ringan namun akurat.
- **Interactive GUI:** Antarmuka web sederhana menggunakan **Streamlit**.
- **Visualisasi:** Grafik probabilitas keyakinan model menggunakan **Altair** (Hijau vs Merah).
- **Preprocessing Otomatis:** Menangani *case folding*, pembersihan simbol, dan normalisasi kata gaul (*typo/slang*).

## 🖼️ Tampilan Aplikasi

| Halaman Utama | Hasil Analisis |
|-------------|-------------|
| ![Home Screen](screenshots/ss1.png) | ![Result Screen](screenshots/ss2.png) |

> *Catatan: Simpan screenshot aplikasi Anda di folder `screenshots/` dengan nama `ss1.png` dan `ss2.png`.*

## 📁 Struktur Proyek
analisis-sentimen-marketplace/ 
│ 
├── app.py # File utama aplikasi Streamlit 
├── model_sentimen_v2.h5 # Model Deep Learning yang sudah dilatih (Otak AI) ├── tokenizer_v2.pickle # Tokenizer untuk memproses teks (Kamus Bahasa)
├── requirements.txt # Daftar library yang dibutuhkan 
└── README.md # Dokumentasi proyek

## 🛠 Cara Menjalankan (Local)

Pastikan Python sudah terinstall di komputer Anda.

**1. Clone Repository**
```bash
git clone [https://github.com/ilham168/analisis-sentimen-sederhana.git](https://github.com/ilham168/analisis-sentimen-sederhana.git)
cd analisis-sentimen-sederhana
2. Install Library Sangat disarankan menggunakan virtual environment. Install dependensi dengan perintah:

Bash

pip install -r requirements.txt
3. Jalankan Aplikasi Gunakan perintah berikut untuk memulai server Streamlit:

Bash

python -m streamlit run app.py
Aplikasi akan otomatis terbuka di browser pada alamat: http://localhost:8501

🧠 Tentang Model
-Model dilatih menggunakan TensorFlow/Keras dengan spesifikasi:

-Preprocessing: Pembersihan teks & Normalisasi Slang.

-Arsitektur: - Embedding Layer: Untuk memahami makna kata.

-GlobalAveragePooling1D: Untuk efisiensi komputasi.

-Dense Layer (ReLU & Sigmoid): Untuk klasifikasi biner.

-Akurasi: ~80% pada data validasi.

💡 Pengembangan Selanjutnya
[ ] Menambahkan dataset yang lebih besar untuk meningkatkan akurasi.

[ ] Implementasi model transformer (IndoBERT) untuk performa lebih tinggi.

[ ] Fitur upload file CSV untuk analisis massal.

[ ] Deploy ke Streamlit Cloud.

📚 Lisensi
MIT License — Silakan gunakan untuk tujuan edukasi dan pengembangan.