import streamlit as st
import pickle
import re

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen JAKI", page_icon="📊")
st.title("Prediksi Sentimen Ulasan Aplikasi JAKI")
st.write("Masukkan teks ulasan untuk memprediksi sentimen (Positif atau Negatif).")

# 2. Memuat Model dan Vektorizer
@st.cache_resource
def muat_model():
    with open('model_sentimen_terbaik.pkl', 'rb') as file_model:
        model = pickle.load(file_model)
    with open('vektorizer_tfidf.pkl', 'rb') as file_vektor:
        vektorizer = pickle.load(file_vektor)
    return model, vektorizer

model, vektorizer = muat_model()

# 3. Fungsi Pembersihan Teks (Sesuai tahap preprocessing)
def bersihkan_teks_input(teks):
    teks = teks.lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    # Catatan: Anda dapat menambahkan integrasi Sastrawi di sini jika diperlukan
    return teks

# 4. Antarmuka Pengguna (Input)
teks_pengguna = st.text_area("Tulis ulasan di sini:", height=150)

# 5. Tombol Prediksi
if st.button("Prediksi Sentimen"):
    if teks_pengguna.strip() == "":
        st.warning("Teks ulasan tidak boleh kosong. Silakan ketik sesuatu.")
    else:
        # Proses Pembersihan
        teks_bersih = bersihkan_teks_input(teks_pengguna)
        
        # Proses Vektorisasi
        teks_vektor = vektorizer.transform([teks_bersih])
        
        # Proses Prediksi
        hasil_prediksi = model.predict(teks_vektor)
        
        # Menampilkan Hasil
        st.subheader("Hasil Prediksi:")
        if hasil_prediksi[0] == 1:
            st.success("Sentimen: POSITIF (Rating 4-5)")
        else:
            st.error("Sentimen: NEGATIF (Rating 1-2)")
