
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import os

# Konfigurasi Path 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")

# daftar fitur bawaan dataset
FITUR_BAWAAN = [
    'url_length', 'has_ip_address', 'dot_count', 'https_flag',
    'url_entropy', 'token_count', 'subdomain_count', 'query_param_count',
    'tld_length', 'path_length', 'has_hyphen_in_domain', 'number_of_digits',
    'tld_popularity', 'suspicious_file_extension', 'domain_name_length',
    'percentage_numeric_chars'
]

# daftar fitur tambahan 
FITUR_KUSTOM = ['http_in_domain']


SEMUA_FITUR = FITUR_BAWAAN + FITUR_KUSTOM


def ekstrak_domain(url):
    """
    Mengekstrak nama domain dari sebuah URL string.

    Proses:
      1. Menghapus skema protokol ('http://' atau 'https://').
      2. Mengambil bagian sebelum '/' pertama (yaitu domain + sub-domain).
      3. Menghapus port jika ada (bagian setelah ':').

    Parameter:
        url (str): String URL lengkap.

    Returns:
        str: Nama domain yang telah diekstrak (lowercase).

    Contoh:
        'https://www.example.com/path' -> 'www.example.com'
        'http://https-secure.pay-pal.com/login' -> 'https-secure.pay-pal.com'
    """
    url = str(url).lower().strip()

    # Hapus skema protokol utama
    if url.startswith('https://'):
        url_tanpa_skema = url[8:]
    elif url.startswith('http://'):
        url_tanpa_skema = url[7:]
    else:
        url_tanpa_skema = url

    # Ambil bagian domain (sebelum '/' pertama)
    domain = url_tanpa_skema.split('/')[0]

    # Hapus port jika ada (misal: example.com:8080)
    domain = domain.split(':')[0]

    return domain








def deteksi_http_in_domain(url):
    """
    Mendeteksi apakah string 'http' atau 'https' muncul di dalam nama domain
    URL, BUKAN sebagai bagian dari skema protokol utama.

    Taktik pengelabuan:
      Penjahat siber sering menyelipkan kata 'http' atau 'https' di dalam
      sub-domain atau nama domain untuk mengelabui pengguna agar mengira
      URL tersebut aman.

    Contoh URL mencurigakan:
      - https://https-secure.pay-pal.com/login  -> http_in_domain = 1
      - http://http.banking.example.com/auth     -> http_in_domain = 1
      - https://www.google.com/                  -> http_in_domain = 0

    Parameter:
        url (str): String URL lengkap.

    Returns:
        int: 1 jika 'http' atau 'https' ditemukan di dalam domain, 0 jika tidak.
    """
    domain = ekstrak_domain(url)

    # Periksa apakah string 'http' atau 'https' ada di dalam nama domain
    # (setelah skema protokol utama sudah dihapus oleh fungsi ekstrak_domain)
    if 'https' in domain or 'http' in domain:
        return 1
    return 0


def tambah_fitur_kustom(df):
    """
    Menambahkan fitur leksikal kustom ke dalam DataFrame:
      1. http_in_domain: Deteksi binary apakah 'http'/'https' ada di domain.

    Parameter:
        df (pd.DataFrame): DataFrame hasil penyeimbangan data (Tahap 2).

    Returns:
        pd.DataFrame: DataFrame dengan 2 kolom fitur baru ditambahkan.
    """
    print("=" * 70)
    print("TAHAP 3: FEATURE ENGINEERING (Bab 3.6)")
    print("=" * 70)

    print(f"\n[INFO] Jumlah fitur bawaan  : {len(FITUR_BAWAAN)}")
    print(f"[INFO] Fitur kustom baru   : {FITUR_KUSTOM}")





    # --- Fitur 2: http_in_domain ---
    # Mendeteksi keberadaan string 'http' atau 'https' di dalam nama domain.
    df['http_in_domain'] = df['URL'].apply(deteksi_http_in_domain)
    print(f"\n  [v] Fitur 'http_in_domain' berhasil ditambahkan.")
    jumlah_positif = df['http_in_domain'].sum()
    persen_positif = (jumlah_positif / len(df)) * 100
    print(f"      Jumlah URL dengan http/https di domain: "
          f"{jumlah_positif:,} ({persen_positif:.2f}%)")

    return df


def verifikasi_fitur(df):
    """
    Memverifikasi bahwa total fitur prediktor pada DataFrame sudah
    berjumlah 17 (16 fitur bawaan + 1 fitur kustom).

    Parameter:
        df (pd.DataFrame): DataFrame setelah penambahan fitur kustom.

    Returns:
        bool: True jika verifikasi berhasil, False jika gagal.
    """
    # Kolom yang bukan fitur prediktor
    kolom_non_fitur = ['URL', 'ClassLabel']
    fitur_aktual = [col for col in df.columns if col not in kolom_non_fitur]

    print(f"\n[VERIFIKASI FITUR]")
    print(f"  Jumlah fitur prediktor : {len(fitur_aktual)}")
    print(f"  Target                 : {len(SEMUA_FITUR)}")

    # Cek apakah semua fitur yang diharapkan ada
    fitur_hilang = [f for f in SEMUA_FITUR if f not in fitur_aktual]
    if fitur_hilang:
        print(f"  [!] GAGAL: Fitur berikut tidak ditemukan: {fitur_hilang}")
        return False

    if len(fitur_aktual) == len(SEMUA_FITUR):
        print(f"  [v] BERHASIL: Total 17 fitur prediktor terverifikasi.")
        print(f"      Daftar fitur: {fitur_aktual}")
        return True
    else:
        print(f"  [!] GAGAL: Jumlah fitur tidak sesuai ({len(fitur_aktual)} != 17)")
        return False


def buat_visualisasi_distribusi(df):
    """
    Membuat visualisasi Grouped Bar Chart yang menampilkan distribusi
    fitur kustom (http_in_domain) berdasarkan kelas
    (Phishing vs Legitimate).

    Visualisasi disimpan sebagai file gambar di folder OUTPUT/.

    Parameter:
        df (pd.DataFrame): DataFrame dengan fitur kustom dan ClassLabel.
    """
    print("\n[VISUALISASI] Membuat Grouped Bar Chart distribusi fitur kustom...")

    # Pastikan folder OUTPUT ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Konfigurasi visual
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle('Distribusi Fitur Kustom berdasarkan Kelas',
                 fontsize=16, fontweight='bold', y=1.02)

    cmap = plt.get_cmap('coolwarm')
    warna_phishing = cmap(0.9)      # Warm color for Phishing
    warna_legitimate = cmap(0.1)    # Cool color for Legitimate
    label_kelas = {0: 'Phishing', 1: 'Legitimate'}

    df_phishing = df[df['ClassLabel'] == 0]
    df_legitimate = df[df['ClassLabel'] == 1]

    # =====================================================================
    # Chart 1: Distribusi http_in_domain (Grouped Bar Chart)
    # =====================================================================
    ax2 = axes

    # http_in_domain hanya memiliki 2 nilai: 0 (Tidak) dan 1 (Ya)
    labels_http = ['0 (Tidak Ada)', '1 (Ada)']

    http_phishing = df_phishing['http_in_domain'].value_counts().reindex(
        [0, 1], fill_value=0)
    http_legitimate = df_legitimate['http_in_domain'].value_counts().reindex(
        [0, 1], fill_value=0)

    x2 = np.arange(len(labels_http))
    lebar_bar = 0.35

    bar3 = ax2.bar(x2 - lebar_bar / 2, http_phishing.values, lebar_bar,
                   label='Phishing', color=warna_phishing, edgecolor='white',
                   alpha=0.85)
    bar4 = ax2.bar(x2 + lebar_bar / 2, http_legitimate.values, lebar_bar,
                   label='Legitimate', color=warna_legitimate, edgecolor='white',
                   alpha=0.85)

    ax2.set_xlabel('Keberadaan http/https di Domain', fontsize=12,
                   fontweight='bold')
    ax2.set_ylabel('Jumlah Sampel', fontsize=12, fontweight='bold')
    ax2.set_title('Distribusi Fitur: http_in_domain', fontsize=14,
                  fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_http)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Tambahkan label angka di atas setiap bar
    for bar in bar3:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{int(height):,}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8)
    for bar in bar4:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{int(height):,}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8)

    # Simpan chart
    plt.tight_layout()
    path_gambar = os.path.join(OUTPUT_DIR, "distribusi_fitur_kustom.png")
    fig.savefig(path_gambar, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"  [v] Visualisasi disimpan ke: {path_gambar}")


def jalankan_feature_engineering(df):
    """
    Fungsi utama yang mengorkestrasi seluruh proses Feature Engineering.
    Memanggil fungsi-fungsi di atas secara berurutan.

    Parameter:
        df (pd.DataFrame): DataFrame hasil penyeimbangan data (Tahap 2).

    Returns:
        pd.DataFrame: DataFrame dengan 17 fitur prediktor (16 bawaan + 1 kustom).
    """
    # 1. Tambahkan fitur kustom
    df = tambah_fitur_kustom(df)

    # 2. Verifikasi jumlah fitur
    verifikasi_fitur(df)

    # 3. Buat visualisasi distribusi
    buat_visualisasi_distribusi(df)

    # 4. Simpan dataset modifikasi 
    path_kustom = os.path.join(BASE_DIR, "DATASET", "dataset_fitur_kustom.csv")
    df.to_csv(path_kustom, index=False)
    print(f"\n  [v] Dataset kustom disimpan ke: {path_kustom}")

    # 5. Tampilkan ringkasan akhir
    print("\n" + "-" * 70)
    print("  RINGKASAN FEATURE ENGINEERING")
    print("-" * 70)
    print(f"  Fitur bawaan  : {len(FITUR_BAWAAN)} fitur")
    print(f"  Fitur kustom  : {len(FITUR_KUSTOM)} fitur ({', '.join(FITUR_KUSTOM)})")
    print(f"  Total fitur   : {len(SEMUA_FITUR)} fitur prediktor")
    print(f"  Total sampel  : {len(df):,} baris")
    print("-" * 70)

    print(f"\n[v] Tahap 3: Feature Engineering selesai.\n")
    return df


# --- Eksekusi Langsung ---
# Blok ini memungkinkan pengujian modul secara independen.
# Menjalankan Tahap 1 & 2 terlebih dahulu, kemudian Tahap 3.
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

    # Jalankan Tahap 3: Feature Engineering
    df_final = jalankan_feature_engineering(df_seimbang)

    # Tampilkan 5 baris pertama dengan fitur baru
    print("\n--- Sampel Data dengan Fitur Kustom ---")
    print(df_final[['URL', 'http_in_domain', 'ClassLabel']].head(10))
    print(f"\nDimensi akhir: {df_final.shape}")
