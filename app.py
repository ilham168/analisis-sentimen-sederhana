import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import re
import altair as alt  # <--- Kita tambah ini untuk grafik yang lebih canggih
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Sentimen Marketplace", layout="centered")

st.title("🛍️ Analisis Sentimen Ulasan")
st.write("Masukkan ulasan produk, dan AI akan mendeteksi apakah itu Positif atau Negatif.")

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model_assets():
    model = tf.keras.models.load_model('model_sentimen_v2.h5')
    with open('tokenizer_v2.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_model_assets()
    st.success("Model AI berhasil dimuat!", icon="✅")
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
    st.stop()

# --- 3. FUNGSI PREPROCESSING ---
key_norm = {"yg": "yang", "gan": "juragan", "bgt": "banget", "gk": "tidak"} 

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [key_norm[word] if word in key_norm else word for word in words]
    text = " ".join(words)
    return text

# --- 4. USER INTERFACE ---
input_text = st.text_area("Tulis ulasan kamu di sini:", height=100, placeholder="Contoh: Barangnya bagus banget, pengiriman cepat!")

if st.button("🔍 Analisis Sekarang"):
    if input_text.strip() == "":
        st.warning("Mohon isi ulasan terlebih dahulu.")
    else:
        with st.spinner('Sedang menganalisis...'):
            # Preprocessing & Prediksi
            text_clean = clean_text(input_text)
            max_length = 100 
            seq = tokenizer.texts_to_sequences([text_clean])
            padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
            
            prediction = model.predict(padded, verbose=0)[0][0]
            
            # Hasil
            persen_positif = prediction
            persen_negatif = 1 - prediction
            
            if persen_positif > 0.5:
                st.success(f"### Hasil: POSITIF 😊 (Keyakinan: {persen_positif:.2%})")
            else:
                st.error(f"### Hasil: NEGATIF 😠 (Keyakinan: {persen_negatif:.2%})")
            
            st.divider()
            
            # --- BAGIAN GRAFIK YANG DIPERBAIKI ---
            st.subheader("📊 Grafik Probabilitas")
            
            chart_data = pd.DataFrame({
                "Sentimen": ["Positif", "Negatif"],
                "Nilai": [persen_positif, persen_negatif]
            })
            
            # Menggunakan Altair agar warna Hijau/Merah akurat
            c = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Sentimen', sort=None),
                y='Nilai',
                color=alt.Color('Sentimen', scale=alt.Scale(
                    domain=['Positif', 'Negatif'],
                    range=['#4CAF50', '#FF5252']  # Hijau dan Merah
                )),
                tooltip=['Sentimen', alt.Tooltip('Nilai', format='.2%')]
            ).properties(height=300)

            st.altair_chart(c, use_container_width=True)
            
            with st.expander("Lihat Detail Teknis"):
                st.write(f"Teks Bersih: {text_clean}")
                st.write(f"Raw Prediction Score: {prediction}")