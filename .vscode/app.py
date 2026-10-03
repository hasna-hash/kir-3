import pandas as pd
import joblib
bundle = joblib.load("model/decision_tree.joblib")
#  =====================================================================
# 1. Load model yang sudah dibuat di STEP 1
# =====================================================================
bundle = joblib.load("model/decision_tree.joblib")
model = bundle["model"]
features = bundle["features"]
# PENTING: jalankan cek_model.py dulu, lalu bandingkan urutan di bawah
# ini dengan urutan yang tercetak. Kalau nama fiturnya beda, sesuaikan
# nama variabel/label di bawah (bukan urutannya, itu sudah otomatis).

# =====================================================================
# 2. Judul Web
# =====================================================================
import streamlit as st
st.title("🌱 Dashboard Kelembapan Tanaman Kangkung")
st.write("Masukkan data kondisi lingkungan untuk memprediksi status penyiraman.")

# =====================================================================
# 3. Buat Input Form — 6 fitur, SESUAI jumlah & urutan yang dipakai model
# =====================================================================
kelembapan_tanah = st.number_input("Kelembapan Tanah (%)", min_value=0.0, max_value=100.0, value=40.0)
suhu_udara = st.number_input("Suhu Udara (°C)", min_value=0.0, max_value=60.0, value=28.0)
kelembapan_udara = st.number_input("Kelembapan Udara (%)", min_value=0.0, max_value=100.0, value=60.0)
intensitas_cahaya = st.number_input("Intensitas Cahaya (Lux)", min_value=0.0, value=10000.0)
waktu_pengukuran = st.number_input("Waktu Pengukuran (Jam, 0-23)", min_value=0, max_value=23, value=12, step=1)
waktu_sejak_disiram = st.number_input("Waktu Sejak Disiram Terakhir (Jam)", min_value=0.0, value=6.0)

# =====================================================================
# 4. Fungsi bantu: menentukan status tanaman dari kelembapan tanah
# =====================================================================
def tentukan_status_tanaman(kelembapan):
    """
    Menerjemahkan angka kelembapan tanah menjadi status yang mudah
    dipahami. Batas angka ini bisa disesuaikan dengan kondisi tanaman
    kangkung kalian (misalnya berdasarkan referensi jurnal/data lapangan).
    """
    if kelembapan < 30:
        return "🥀 Kering", "warning"
    elif kelembapan <= 70:
        return "🌿 Ideal", "success"
    else:
        return "💧 Terlalu Basah", "info"


# =====================================================================
# 5. Fungsi bantu: mengubah hasil prediksi jadi rekomendasi aksi
# =====================================================================
def buat_rekomendasi(prediksi):
    """Mengubah label prediksi mentah jadi kalimat rekomendasi aksi."""
    if prediksi == "Perlu Disiram":
        return "💦 Segera siram tanaman sekarang."
    else:
        return "✅ Tidak perlu disiram, kondisi masih cukup air."


# =====================================================================
# 6. Tombol Prediksi
# =====================================================================
if st.button("Cek Status Penyiraman"):
    # Gabungkan input ke format DataFrame sesuai fitur model.
    # Urutan list di bawah HARUS sama dengan urutan `features` di atas.
    input_data = pd.DataFrame(
        [[
            kelembapan_tanah,
            suhu_udara,
            kelembapan_udara,
            intensitas_cahaya,
            waktu_pengukuran,
            waktu_sejak_disiram,
        ]],
        columns=features,
    )

    # Lakukan prediksi
    prediksi = model.predict(input_data)[0]

    # Tingkat keyakinan model, dalam persen
    proba = model.predict_proba(input_data)[0]
    keyakinan = round(max(proba) * 100, 1)

    # Terjemahkan hasil mentah jadi status & rekomendasi yang mudah dibaca
    status_tanaman, jenis_status = tentukan_status_tanaman(kelembapan_tanah)
    rekomendasi = buat_rekomendasi(prediksi)

    # =================================================================
    # 7. Tampilkan semua hasil dalam beberapa kolom biar rapi
    # =================================================================
    st.subheader("📋 Hasil Analisis")

    kolom1, kolom2 = st.columns(2)
    with kolom1:
        st.metric("Kelembapan Tanah Saat Ini", f"{kelembapan_tanah}%")
    with kolom2:
        st.metric("Keyakinan Model", f"{keyakinan}%")

    # Status tanaman (kering / ideal / terlalu basah)
    if jenis_status == "warning":
        st.warning(f"Status Tanaman: {status_tanaman}")
    elif jenis_status == "success":
        st.success(f"Status Tanaman: {status_tanaman}")
    else:
        st.info(f"Status Tanaman: {status_tanaman}")

    # Hasil prediksi model (merah/hijau)
    if prediksi == "Perlu Disiram":
        st.error(f"🔴 Prediksi Model: **{prediksi}**")
    else:
        st.success(f"🟢 Prediksi Model: **{prediksi}**")

    # Rekomendasi aksi yang jelas untuk pengguna
    st.markdown(f"### Rekomendasi: {rekomendasi}")