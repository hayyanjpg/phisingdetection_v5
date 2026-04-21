# ==============================================================================
# MODUL 04: SPLIT DATA & SCALING (Bab 3.7 - 3.8)
# ==============================================================================
# Deskripsi:
#   Modul ini bertanggung jawab untuk membagi dataset menjadi data latih
#   (training) dan data uji (testing), serta melakukan standardisasi fitur.
#   Proses yang dilakukan:
#     1. Pemisahan fitur prediktor (X) dan label target (y).
#        Kolom 'URL' TIDAK diikutsertakan ke dalam fitur prediktor.
#     2. Pembagian data menggunakan Stratified Sampling dengan rasio 80:20.
#        Stratified Sampling memastikan proporsi kelas tetap terjaga
#        pada data latih maupun data uji.
#     3. Standardisasi fitur menggunakan StandardScaler.
#        PENTING: Scaler di-fit HANYA pada data latih (X_train), kemudian
#        di-transform pada data latih DAN data uji. Hal ini untuk menghindari
#        data leakage (kebocoran informasi dari data uji ke proses pelatihan).
#     4. Penyimpanan objek scaler dan daftar nama fitur ke folder OUTPUT/
#        dalam format pickle (.pkl) agar dapat digunakan kembali saat
#        inferensi pada aplikasi web.
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Konfigurasi Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")

# --- Konfigurasi Parameter ---
TEST_SIZE = 0.20        # Rasio data uji: 20%
RANDOM_STATE = 42       # Seed untuk reproduksibilitas hasil


def pisahkan_fitur_dan_label(df):
    """
    Memisahkan DataFrame menjadi matriks fitur prediktor (X) dan
    vektor label target (y).

    Kolom 'URL' tidak diikutsertakan ke dalam fitur prediktor karena
    merupakan data mentah berupa string yang tidak dapat langsung
    digunakan oleh algoritma machine learning.

    Parameter:
        df (pd.DataFrame): DataFrame lengkap dengan fitur, URL, dan ClassLabel.

    Returns:
        tuple: (X, y, feature_names)
            - X (pd.DataFrame): Matriks fitur prediktor (18 kolom).
            - y (pd.Series): Vektor label target (ClassLabel).
            - feature_names (list): Daftar nama fitur prediktor.
    """
    print("=" * 70)
    print("TAHAP 4: SPLIT DATA & SCALING (Bab 3.7 - 3.8)")
    print("=" * 70)

    # Kolom yang TIDAK digunakan sebagai fitur prediktor
    kolom_eksklusi = ['URL', 'ClassLabel']

    # Ambil seluruh kolom kecuali URL dan ClassLabel sebagai fitur prediktor
    feature_names = [col for col in df.columns if col not in kolom_eksklusi]
    X = df[feature_names].copy()
    y = df['ClassLabel'].copy()

    print(f"\n[PEMISAHAN FITUR & LABEL]")
    print(f"  Jumlah fitur prediktor (X) : {X.shape[1]}")
    print(f"  Daftar fitur               : {feature_names}")
    print(f"  Jumlah sampel              : {len(X):,}")
    print(f"  Distribusi label (y)       :")
    for label in sorted(y.unique()):
        nama = "Phishing" if label == 0 else "Legitimate"
        jumlah = (y == label).sum()
        persen = (jumlah / len(y)) * 100
        print(f"    Kelas {label} ({nama:>10}): {jumlah:>7,} ({persen:.2f}%)")

    return X, y, feature_names


def split_stratified(X, y):
    """
    Membagi data menjadi set latih (training) dan set uji (testing)
    menggunakan Stratified Sampling.

    Stratified Sampling menjamin bahwa proporsi setiap kelas pada
    data latih dan data uji tetap konsisten dengan proporsi pada
    dataset keseluruhan. Hal ini penting terutama pada masalah
    klasifikasi untuk menghindari bias distribusi kelas.

    Parameter:
        X (pd.DataFrame): Matriks fitur prediktor.
        y (pd.Series): Vektor label target.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\n[SPLIT DATA - Stratified Sampling]")
    print(f"  Rasio               : {int((1-TEST_SIZE)*100)}% Training : "
          f"{int(TEST_SIZE*100)}% Testing")
    print(f"  Random state        : {RANDOM_STATE}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Stratified Sampling berdasarkan distribusi label
    )

    print(f"\n  Hasil Split:")
    print(f"    X_train : {X_train.shape[0]:>7,} sampel x {X_train.shape[1]} fitur")
    print(f"    X_test  : {X_test.shape[0]:>7,} sampel x {X_test.shape[1]} fitur")
    print(f"    y_train : {y_train.shape[0]:>7,} sampel")
    print(f"    y_test  : {y_test.shape[0]:>7,} sampel")

    # Verifikasi bahwa proporsi kelas terjaga setelah split
    print(f"\n  Verifikasi Stratified Sampling:")
    for label in sorted(y.unique()):
        nama = "Phishing" if label == 0 else "Legitimate"
        persen_train = (y_train == label).sum() / len(y_train) * 100
        persen_test = (y_test == label).sum() / len(y_test) * 100
        print(f"    Kelas {label} ({nama:>10}): "
              f"Train={persen_train:.2f}% | Test={persen_test:.2f}%")

    return X_train, X_test, y_train, y_test


def standardisasi_fitur(X_train, X_test, feature_names):
    """
    Melakukan standardisasi (StandardScaler) pada fitur prediktor.

    StandardScaler mentransformasi setiap fitur sehingga memiliki:
      - Mean (rata-rata) = 0
      - Standard deviation (simpangan baku) = 1

    PENTING - Pencegahan Data Leakage:
      Scaler di-fit (dihitung mean & std) HANYA pada data latih (X_train).
      Kemudian, transformasi diterapkan pada KEDUA set data (X_train & X_test)
      menggunakan parameter yang SAMA dari data latih.
      Hal ini mencegah kebocoran informasi dari data uji ke proses pelatihan.

    Parameter:
        X_train (pd.DataFrame): Fitur data latih.
        X_test (pd.DataFrame): Fitur data uji.
        feature_names (list): Daftar nama fitur prediktor.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
            - X_train_scaled (np.ndarray): Fitur data latih yang telah di-scale.
            - X_test_scaled (np.ndarray): Fitur data uji yang telah di-scale.
            - scaler (StandardScaler): Objek scaler yang telah di-fit.
    """
    print(f"\n[STANDARDISASI - StandardScaler]")

    scaler = StandardScaler()

    # Fit scaler HANYA pada data latih
    # (menghitung mean dan standard deviation dari data latih)
    scaler.fit(X_train)
    print("  [v] Scaler di-fit pada data latih (X_train).")

    # Transform kedua set data menggunakan parameter dari data latih
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  [v] X_train dan X_test telah di-transform.")

    # Tampilkan statistik hasil scaling (verifikasi)
    print(f"\n  Verifikasi Hasil Scaling (X_train):")
    print(f"    Mean   : {X_train_scaled.mean(axis=0).mean():.6f} "
          f"(diharapkan ~ 0.0)")
    print(f"    Std    : {X_train_scaled.std(axis=0).mean():.6f} "
          f"(diharapkan ~ 1.0)")

    print(f"\n  Verifikasi Hasil Scaling (X_test):")
    print(f"    Mean   : {X_test_scaled.mean(axis=0).mean():.6f}")
    print(f"    Std    : {X_test_scaled.std(axis=0).mean():.6f}")

    return X_train_scaled, X_test_scaled, scaler


def simpan_artefak(scaler, feature_names):
    """
    Menyimpan objek scaler dan daftar nama fitur ke folder OUTPUT/
    dalam format pickle (.pkl).

    Artefak ini diperlukan saat inferensi pada aplikasi web agar:
      - Fitur baru diekstrak dalam urutan yang sama.
      - Scaling menggunakan parameter (mean & std) yang sama dengan
        yang digunakan saat pelatihan model.

    Parameter:
        scaler (StandardScaler): Objek scaler yang telah di-fit.
        feature_names (list): Daftar nama fitur prediktor.
    """
    print(f"\n[PENYIMPANAN ARTEFAK]")

    # Pastikan folder OUTPUT ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Simpan objek StandardScaler
    path_scaler = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(scaler, path_scaler)
    print(f"  [v] Scaler disimpan ke: {path_scaler}")

    # Simpan daftar nama fitur
    path_feature_names = os.path.join(OUTPUT_DIR, "feature_names.pkl")
    joblib.dump(feature_names, path_feature_names)
    print(f"  [v] Feature names disimpan ke: {path_feature_names}")

    # Verifikasi: muat kembali dan cek
    scaler_loaded = joblib.load(path_scaler)
    features_loaded = joblib.load(path_feature_names)
    print(f"\n  Verifikasi Artefak:")
    print(f"    Scaler type   : {type(scaler_loaded).__name__}")
    print(f"    Jumlah fitur  : {len(features_loaded)}")
    print(f"    Fitur tersimpan: {features_loaded}")


def simpan_data_scaled(X_train_scaled, X_test_scaled, y_train, y_test,
                       feature_names):
    """
    Menyimpan data hasil scaling ke file CSV untuk keperluan dokumentasi
    pada Bab 4 skripsi (lampiran).

    File yang dihasilkan:
      - X_train_scaled.csv : Data latih yang telah di-standardisasi.
      - X_test_scaled.csv  : Data uji yang telah di-standardisasi.

    Setiap file menyertakan nama fitur sebagai header kolom dan
    kolom ClassLabel di akhir agar memudahkan analisis.

    Parameter:
        X_train_scaled (np.ndarray): Matriks fitur data latih (ter-scaled).
        X_test_scaled (np.ndarray): Matriks fitur data uji (ter-scaled).
        y_train (pd.Series): Label data latih.
        y_test (pd.Series): Label data uji.
        feature_names (list): Daftar nama fitur prediktor.
    """
    print(f"\n[PENYIMPANAN DATA HASIL SCALING]")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Buat DataFrame dari data scaled dengan nama kolom fitur
    df_train = pd.DataFrame(X_train_scaled, columns=feature_names)
    df_train['ClassLabel'] = y_train.values

    df_test = pd.DataFrame(X_test_scaled, columns=feature_names)
    df_test['ClassLabel'] = y_test.values

    # Simpan ke CSV
    path_train = os.path.join(OUTPUT_DIR, "X_train_scaled.csv")
    path_test = os.path.join(OUTPUT_DIR, "X_test_scaled.csv")

    df_train.to_csv(path_train, index=False)
    df_test.to_csv(path_test, index=False)

    print(f"  [v] Data latih tersimpan  : {path_train}")
    print(f"      Dimensi: {df_train.shape[0]:,} baris x {df_train.shape[1]} kolom")
    print(f"  [v] Data uji tersimpan   : {path_test}")
    print(f"      Dimensi: {df_test.shape[0]:,} baris x {df_test.shape[1]} kolom")

    # Tampilkan sampel 5 baris pertama data latih sebagai preview
    print(f"\n  Preview Data Latih (5 baris pertama, 5 fitur pertama):")
    preview = df_train.iloc[:5, :5]
    for idx, row in preview.iterrows():
        vals = "  ".join([f"{v:>8.4f}" for v in row.values])
        print(f"    [{idx}] {vals}")
    print(f"    ... ({df_train.shape[1] - 5} kolom lainnya tidak ditampilkan)")

def buat_visualisasi_split(X_train, X_test):
    """
    Membuat Pie Chart untuk merepresentasikan proporsi training vs testing data.
    """
    print("\n[VISUALISASI] Membuat grafik proporsi Split Data...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path_gambar = os.path.join(OUTPUT_DIR, "grafik_splitting.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    
    labels = ['Training Set (80%)', 'Testing Set (20%)']
    sizes = [len(X_train), len(X_test)]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(0.3), cmap(0.8)]

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.axis('equal') 
    plt.title('Proporsi Pembagian Data Latih dan Uji', fontsize=14, fontweight='bold', y=1.05)

    fig.tight_layout()
    fig.savefig(path_gambar, dpi=150, facecolor='white')
    plt.close(fig)
    print(f"  [v] Visualisasi disimpan ke: {path_gambar}")


def jalankan_split_scaling(df):
    """
    Fungsi utama yang mengorkestrasi seluruh proses split data dan scaling.
    Memanggil fungsi-fungsi di atas secara berurutan.

    Parameter:
        df (pd.DataFrame): DataFrame hasil Feature Engineering (Tahap 3),
                           berisi kolom URL, 18 fitur prediktor, dan ClassLabel.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    """
    # 1. Pisahkan fitur (X) dan label (y)
    X, y, feature_names = pisahkan_fitur_dan_label(df)

    # 2. Split data dengan Stratified Sampling (80:20)
    X_train, X_test, y_train, y_test = split_stratified(X, y)

    # 3. Standardisasi fitur (fit pada train, transform pada train & test)
    X_train_scaled, X_test_scaled, scaler = standardisasi_fitur(
        X_train, X_test, feature_names)

    # 4. Simpan scaler dan feature_names ke OUTPUT/
    simpan_artefak(scaler, feature_names)

    # 5. Simpan data hasil scaling ke CSV (untuk dokumentasi Bab 4)
    simpan_data_scaled(X_train_scaled, X_test_scaled,
                       y_train, y_test, feature_names)

    # 6. Buat visualisasi
    buat_visualisasi_split(X_train, X_test)

    # 7. Tampilkan ringkasan akhir
    print("\n" + "-" * 70)
    print("  RINGKASAN SPLIT & SCALING")
    print("-" * 70)
    print(f"  Total sampel      : {len(X):,}")
    print(f"  Training set      : {X_train_scaled.shape[0]:,} sampel "
          f"({X_train_scaled.shape[0]/len(X)*100:.0f}%)")
    print(f"  Testing set       : {X_test_scaled.shape[0]:,} sampel "
          f"({X_test_scaled.shape[0]/len(X)*100:.0f}%)")
    print(f"  Jumlah fitur      : {X_train_scaled.shape[1]}")
    print(f"  Metode scaling    : StandardScaler")
    print(f"  Artefak tersimpan : scaler.pkl, feature_names.pkl,")
    print(f"                      X_train_scaled.csv, X_test_scaled.csv")
    print("-" * 70)

    print(f"\n[v] Tahap 4: Split & Scaling selesai.\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


# --- Eksekusi Langsung ---
# Blok ini memungkinkan pengujian modul secara independen.
# Menjalankan Tahap 1, 2, 3 terlebih dahulu, kemudian Tahap 4.
if __name__ == "__main__":
    import importlib.util

    # Muat dan jalankan Tahap 1: Pra-Pemrosesan
    spec1 = importlib.util.spec_from_file_location(
        "preprocessing", os.path.join(BASE_DIR, "01_preprocessing.py"))
    mod1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(mod1)
    df_bersih = mod1.jalankan_preprocessing()

    # Muat dan jalankan Tahap 2: Penyeimbangan Data
    spec2 = importlib.util.spec_from_file_location(
        "balancing", os.path.join(BASE_DIR, "02_balancing.py"))
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    df_seimbang = mod2.jalankan_balancing(df_bersih)

    # Muat dan jalankan Tahap 3: Feature Engineering
    spec3 = importlib.util.spec_from_file_location(
        "feature_engineering", os.path.join(BASE_DIR, "03_feature_engineering.py"))
    mod3 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(mod3)
    df_final = mod3.jalankan_feature_engineering(df_seimbang)

    # Jalankan Tahap 4: Split & Scaling
    X_train, X_test, y_train, y_test, feat_names = jalankan_split_scaling(df_final)

    # Tampilkan dimensi akhir
    print("\n--- Dimensi Data Akhir ---")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")
