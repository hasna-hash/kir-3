"""
=====================================================================
 Script Python: Prediksi Kebutuhan Penyiraman Tanaman
 Algoritma: Decision Tree (scikit-learn)
=====================================================================
Latar belakang (sesuai abstrak penelitian):
Sistem penyiraman sederhana umumnya menggunakan jadwal atau batas
kelembapan tertentu, sehingga belum mempertimbangkan berbagai kondisi
lingkungan yang memengaruhi kebutuhan air tanaman. Script ini
menerapkan Decision Tree untuk memprediksi kebutuhan penyiraman
tanaman (Perlu Disiram / Belum Perlu Disiram) berdasarkan data:
    - Kelembapan_Tanah (%)
    - Suhu_Udara (°C)
    - Kelembapan_Udara (%)
    - Intensitas_Cahaya (lux)
    - Waktu_Pengukuran (jam, 0-23)
    - Waktu_Sejak_Disiram (jam)

Cara install library yang dibutuhkan:
    pip install pandas numpy scikit-learn matplotlib
=====================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# =====================================================================
# 1. PERSIAPAN DATA (DUMMY DATASET)
# =====================================================================
# Karena belum ada data sensor asli, kita buat dataset sintetis dengan
# aturan logika sederhana (bukan acak murni) supaya pola yang dipelajari
# model tetap masuk akal secara agronomi:
# - Tanah kering + udara panas & kering + cahaya terang + lama tidak
#   disiram  -> cenderung PERLU DISIRAM
# - Tanah lembap + udara sejuk & lembap + baru saja disiram
#   -> cenderung BELUM PERLU DISIRAM
def buat_dataset_dummy(n_sampel=500, seed=42):
    rng = np.random.default_rng(seed)

    kelembapan_tanah = rng.uniform(5, 95, n_sampel)          # %
    suhu_udara = rng.uniform(18, 38, n_sampel)               # °C
    kelembapan_udara = rng.uniform(25, 90, n_sampel)         # %
    intensitas_cahaya = rng.uniform(0, 100000, n_sampel)     # lux
    waktu_pengukuran = rng.integers(0, 24, n_sampel)         # jam
    waktu_sejak_disiram = rng.uniform(0, 48, n_sampel)       # jam

    # Skor kebutuhan air: kombinasi kelembapan tanah rendah,
    # suhu tinggi, udara kering, cahaya terang, dan lama sejak disiram.
    skor = (
        (60 - kelembapan_tanah) / 60 * 0.45
        + (suhu_udara - 18) / 20 * 0.20
        + (90 - kelembapan_udara) / 65 * 0.15
        + (intensitas_cahaya / 100000) * 0.10
        + (waktu_sejak_disiram / 48) * 0.10
    )
    noise = rng.normal(0, 0.05, n_sampel)  # sedikit noise agar realistis

    kondisi = np.where(skor + noise > 0.35, "Perlu Disiram", "Belum Perlu Disiram")

    df = pd.DataFrame({
        "Kelembapan_Tanah": kelembapan_tanah.round(2),
        "Suhu_Udara": suhu_udara.round(2),
        "Kelembapan_Udara": kelembapan_udara.round(2),
        "Intensitas_Cahaya": intensitas_cahaya.round(1),
        "Waktu_Pengukuran": waktu_pengukuran,
        "Waktu_Sejak_Disiram": waktu_sejak_disiram.round(2),
        "Kondisi": kondisi,
    })
    return df


dataset = buat_dataset_dummy()
print("Contoh 5 baris dataset:")
print(dataset.head(), "\n")
print("Distribusi label:")
print(dataset["Kondisi"].value_counts(), "\n")


# =====================================================================
# 2. PREPROCESSING DATA
# =====================================================================
FITUR = [
    "Kelembapan_Tanah",
    "Suhu_Udara",
    "Kelembapan_Udara",
    "Intensitas_Cahaya",
    "Waktu_Pengukuran",
    "Waktu_Sejak_Disiram",
]
TARGET = "Kondisi"

X = dataset[FITUR]
y = dataset[TARGET]

# Bagi data menjadi 75% latih dan 25% uji.
# stratify=y menjaga proporsi kelas tetap seimbang di kedua bagian.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Jumlah data latih : {len(X_train)}")
print(f"Jumlah data uji   : {len(X_test)}\n")


# =====================================================================
# 3. PELATIHAN MODEL (DECISION TREE)
# =====================================================================
# max_depth dan min_samples_leaf dibatasi agar model tidak overfitting
# (tidak menghafal data latih, tetap bisa menggeneralisasi data baru).
model = DecisionTreeClassifier(
    max_depth=4,           # kedalaman pohon dibatasi
    min_samples_leaf=10,   # setiap daun minimal berisi 10 sampel
    random_state=42,
)
model.fit(X_train, y_train)
print("Model Decision Tree berhasil dilatih.\n")


# =====================================================================
# 4. EVALUASI MODEL
# =====================================================================
y_pred = model.predict(X_test)

akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi model pada data uji: {akurasi * 100:.2f}%\n")

print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred))


# =====================================================================
# 5. VISUALISASI POHON KEPUTUSAN
# =====================================================================
def tampilkan_pohon_keputusan(model, fitur, simpan_sebagai=None):
    """Menampilkan (dan opsional menyimpan) visualisasi Decision Tree."""
    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=fitur,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Visualisasi Decision Tree - Prediksi Kebutuhan Penyiraman")
    if simpan_sebagai:
        plt.savefig(simpan_sebagai, dpi=150, bbox_inches="tight")
        print(f"Gambar pohon keputusan disimpan sebagai '{simpan_sebagai}'")
    plt.show()


tampilkan_pohon_keputusan(model, FITUR, simpan_sebagai="decision_tree.png")


# =====================================================================
# 6. FUNGSI PREDIKSI DATA BARU
# =====================================================================
def prediksi_penyiraman(
    kelembapan_tanah,
    suhu_udara,
    kelembapan_udara,
    intensitas_cahaya,
    waktu_pengukuran,
    waktu_sejak_disiram,
):
    """
    Memprediksi kebutuhan penyiraman dari satu set nilai sensor baru.

    Parameter:
        kelembapan_tanah      : float, dalam %
        suhu_udara            : float, dalam °C
        kelembapan_udara      : float, dalam %
        intensitas_cahaya     : float, dalam lux
        waktu_pengukuran      : int, jam (0-23)
        waktu_sejak_disiram   : float, dalam jam

    Return:
        dict berisi label prediksi dan tingkat keyakinan model (%).
    """
    data_baru = pd.DataFrame([{
        "Kelembapan_Tanah": kelembapan_tanah,
        "Suhu_Udara": suhu_udara,
        "Kelembapan_Udara": kelembapan_udara,
        "Intensitas_Cahaya": intensitas_cahaya,
        "Waktu_Pengukuran": waktu_pengukuran,
        "Waktu_Sejak_Disiram": waktu_sejak_disiram,
    }])[FITUR]

    label_prediksi = model.predict(data_baru)[0]
    proba = model.predict_proba(data_baru)[0]
    keyakinan = max(proba) * 100

    return {
        "prediksi": label_prediksi,
        "keyakinan_persen": round(keyakinan, 1),
    }


# Contoh penggunaan fungsi prediksi dengan data sensor baru:
if __name__ == "__main__":
    print("\n=== Contoh Prediksi Data Baru ===")

    contoh_1 = prediksi_penyiraman(
        kelembapan_tanah=18,
        suhu_udara=34,
        kelembapan_udara=30,
        intensitas_cahaya=75000,
        waktu_pengukuran=13,
        waktu_sejak_disiram=20,
    )
    print("Kondisi kering & panas siang hari ->", contoh_1)

    contoh_2 = prediksi_penyiraman(
        kelembapan_tanah=75,
        suhu_udara=22,
        kelembapan_udara=80,
        intensitas_cahaya=4000,
        waktu_pengukuran=6,
        waktu_sejak_disiram=2,
    )
    print("Kondisi lembap & baru disiram pagi hari ->", contoh_2)
    # ... (kode load data, buat dataset) ...

model = DecisionTreeClassifier()
model.fit(X_train, y_train)   # <-- model sudah dilatih di sini
# BARU SETELAH INI, kode simpan model kalian dijalankan:
import os
import joblib

os.makedirs("model", exist_ok=True)
data_bundle = {
    "model": model,
    "features": list(X.columns),
}
joblib.dump(data_bundle, "model/decision_tree.joblib")
print("Model dan daftar fitur berhasil disimpan ke 'model/decision_tree.joblib'!")