

import pandas as pd
import os

# --- Konfigurasi Path ---
# Menentukan direktori dasar secara dinamis berdasarkan lokasi file ini.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "DATASET", "url_features_extracted1.csv")

# --- Daftar 16 Fitur Leksikal Bawaan ---
# Fitur-fitur ini merupakan fitur prediktor asli yang terdapat dalam dataset.
FITUR_LEKSIKAL = [
    'url_length',
    'has_ip_address',
    'dot_count',
    'https_flag',
    'url_entropy',
    'token_count',
    'subdomain_count',
    'query_param_count',
    'tld_length',
    'path_length',
    'has_hyphen_in_domain',
    'number_of_digits',
    'tld_popularity',
    'suspicious_file_extension',
    'domain_name_length',
    'percentage_numeric_chars'
]

# Kolom yang akan dipertahankan: URL (untuk referensi), 16 fitur, dan label.
KOLOM_TERPILIH = ['URL'] + FITUR_LEKSIKAL + ['ClassLabel']


def muat_dataset(path=DATASET_PATH):
    """
    Memuat dataset dari file CSV ke dalam DataFrame Pandas.

    Parameter:
        path (str): Lokasi file CSV dataset.

    Returns:
        pd.DataFrame: DataFrame yang berisi seluruh data mentah.
    """
    print("=" * 70)
    print("TAHAP 1: PRA-PEMROSESAN DATA (Bab 3.4)")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"\n[INFO] Dataset berhasil dimuat dari: {path}")
    print(f"[INFO] Dimensi awal dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


def seleksi_fitur(df):
    """
    Melakukan seleksi kolom/fitur yang relevan untuk analisis.
    Hanya mempertahankan kolom URL, 16 fitur leksikal, dan ClassLabel.

    Parameter:
        df (pd.DataFrame): DataFrame mentah.

    Returns:
        pd.DataFrame: DataFrame dengan kolom terpilih.
    """
    df = df[KOLOM_TERPILIH].copy()
    print(f"\n[SELEKSI FITUR] {len(FITUR_LEKSIKAL)} fitur leksikal dipilih.")
    print(f"  Kolom dipertahankan: {list(df.columns)}")
    return df


def tampilkan_statistik(df, judul="STATISTIK DATASET"):
    """
    Menampilkan rangkuman statistik deskriptif dari DataFrame.

    Parameter:
        df (pd.DataFrame): DataFrame yang akan dianalisis.
        judul (str): Judul yang ditampilkan di header statistik.
    """
    print("\n" + "-" * 70)
    print(f"  {judul}")
    print("-" * 70)
    print(f"  Total baris         : {df.shape[0]:,}")
    print(f"  Total kolom         : {df.shape[1]}")
    print(f"  Missing values      : {df.isnull().sum().sum()}")
    print(f"  Data duplikat       : {df.duplicated().sum():,}")

    if 'ClassLabel' in df.columns:
        print(f"\n  Distribusi ClassLabel:")
        distribusi = df['ClassLabel'].value_counts().sort_index()
        for label, jumlah in distribusi.items():
            nama_kelas = "Phishing" if label == 0 else "Legitimate"
            persen = (jumlah / len(df)) * 100
            print(f"    Kelas {int(label)} ({nama_kelas:>10}): {jumlah:>7,} ({persen:.2f}%)")

    if 'URL' in df.columns:
        # Deteksi baris anomali (URL tanpa skema yang valid)
        anomali_offline = df[df['URL'].str.strip().str.lower() == 'offline']
        print(f"  Baris anomali (URL='Offline'): {len(anomali_offline)}")

    if 'subdomain_count' in df.columns:
        anomali_subdomain = df[df['subdomain_count'] == -1]
        print(f"  Baris anomali (subdomain_count=-1): {len(anomali_subdomain)}")

    print("-" * 70)


def bersihkan_data(df):
    """
    Melakukan serangkaian proses pembersihan data, meliputi:
      1. Konversi URL ke lowercase dan penghapusan whitespace di ujung.
      2. Penghapusan baris dengan missing values (NaN).
      3. Penghapusan baris anomali:
         - URL bernilai "offline" (bukan URL valid).
         - subdomain_count bernilai -1 (indikasi URL tanpa skema protokol).
      4. Penghapusan baris duplikat.
      5. Konversi ClassLabel dari float ke integer.

    Parameter:
        df (pd.DataFrame): DataFrame yang telah melewati seleksi fitur.

    Returns:
        pd.DataFrame: DataFrame yang telah dibersihkan.
    """
    jumlah_awal = len(df)
    print("\n[PROSES CLEANING]")

    # --- Langkah 1: Normalisasi URL ---
    # Mengubah seluruh string URL menjadi huruf kecil (lowercase)
    # dan menghapus spasi kosong (whitespace) di awal dan akhir string.
    df['URL'] = df['URL'].astype(str).str.strip().str.lower()
    print("  [v] URL dikonversi ke lowercase dan whitespace dihapus.")

    # --- Langkah 2: Hapus Missing Values ---
    # Menghapus baris yang mengandung nilai kosong (NaN) pada kolom manapun.
    jumlah_na = df.isnull().sum().sum()
    df = df.dropna()
    print(f"  [v] Missing values dihapus: {jumlah_na} nilai NaN ditemukan & dihapus.")

    # --- Langkah 3: Hapus Baris Anomali ---
    # 3a. Menghapus baris dengan URL = "offline" (bukan URL yang valid).
    mask_offline = df['URL'] == 'offline'
    jumlah_offline = mask_offline.sum()
    df = df[~mask_offline]
    print(f"  [v] Baris anomali (URL='Offline') dihapus: {jumlah_offline} baris.")

    # 3b. Menghapus baris dengan subdomain_count = -1.
    #     Nilai -1 mengindikasikan bahwa URL tidak memiliki skema protokol
    #     yang valid (misalnya URL yang dimulai tanpa 'http://' atau 'https://').
    mask_subdomain = df['subdomain_count'] == -1
    jumlah_subdomain = mask_subdomain.sum()
    df = df[~mask_subdomain]
    print(f"  [v] Baris anomali (subdomain_count=-1) dihapus: {jumlah_subdomain} baris.")

    # --- Langkah 4: Hapus Data Duplikat ---
    jumlah_duplikat = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"  [v] Data duplikat dihapus: {jumlah_duplikat:,} baris.")

    # --- Langkah 5: Konversi ClassLabel ke Integer ---
    # Mengubah tipe data ClassLabel dari float64 menjadi int64.
    # Phishing = 0, Legitimate = 1.
    df['ClassLabel'] = df['ClassLabel'].astype(int)
    print("  [v] ClassLabel dikonversi ke tipe Integer (0=Phishing, 1=Legitimate).")

    # --- Ringkasan Cleaning ---
    jumlah_akhir = len(df)
    jumlah_dihapus = jumlah_awal - jumlah_akhir
    print(f"\n  [RINGKASAN] Total baris dihapus: {jumlah_dihapus:,} "
          f"({jumlah_awal:,} -> {jumlah_akhir:,})")

    # Reset index setelah penghapusan baris
    df = df.reset_index(drop=True)

    return df


def jalankan_preprocessing():
    """
    Fungsi utama yang mengorkestrasi seluruh proses pra-pemrosesan data.
    Memanggil fungsi-fungsi di atas secara berurutan dan mengembalikan
    DataFrame yang telah bersih serta siap untuk tahap selanjutnya.

    Returns:
        pd.DataFrame: DataFrame hasil preprocessing.
    """
    # 1. Muat dataset
    df = muat_dataset()

    # 2. Seleksi fitur yang relevan
    df = seleksi_fitur(df)

    # 3. Tampilkan statistik SEBELUM cleaning
    tampilkan_statistik(df, judul="STATISTIK SEBELUM CLEANING")

    # 4. Lakukan proses cleaning
    df = bersihkan_data(df)

    # 5. Tampilkan statistik SESUDAH cleaning
    tampilkan_statistik(df, judul="STATISTIK SESUDAH CLEANING")

    print("\n[v] Tahap 1: Pra-Pemrosesan selesai.\n")
    return df


# --- Eksekusi Langsung ---
# Blok ini hanya dijalankan ketika file dieksekusi secara langsung,
# bukan ketika di-import sebagai modul oleh file lain.
if __name__ == "__main__":
    df_bersih = jalankan_preprocessing()
    print(df_bersih.head(10))
    print(f"\nTipe data:\n{df_bersih.dtypes}")
