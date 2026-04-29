
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Konfigurasi Path 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_SEIMBANG = os.path.join(BASE_DIR, "DATASET", "dataset_raw_seimbang.csv")


def hitung_distribusi_kelas(df):
    """
    Menghitung dan menampilkan distribusi kelas dalam dataset.

    Parameter:
        df (pd.DataFrame): DataFrame dengan kolom 'ClassLabel'.

    Returns:
        dict: Dictionary berisi jumlah sampel per kelas.
              Contoh: {0: 63000, 1: 37000}
    """
    distribusi = df['ClassLabel'].value_counts().to_dict()

    print("\n  Distribusi Kelas Saat Ini:")
    for label in sorted(distribusi.keys()):
        nama = "Phishing" if label == 0 else "Legitimate"
        jumlah = distribusi[label]
        persen = (jumlah / len(df)) * 100
        print(f"    Kelas {label} ({nama:>10}): {jumlah:>7,} sampel ({persen:.2f}%)")
    print(f"    {'-' * 50}")
    print(f"    Total                    : {len(df):>7,} sampel")

    return distribusi


def random_undersampling(df):
    """
    Melakukan Random Undersampling pada kelas mayoritas secara dinamis.

    Proses:
      1. Identifikasi kelas minoritas dan mayoritas secara otomatis.
      2. Ambil sampel acak dari kelas mayoritas sebanyak jumlah kelas minoritas.
      3. Gabungkan kembali kedua kelas sehingga rasio menjadi 50:50.

    Parameter:
        df (pd.DataFrame): DataFrame hasil preprocessing dengan kolom 'ClassLabel'.

    Returns:
        pd.DataFrame: DataFrame dengan distribusi kelas yang seimbang (50:50).
    """
    print("=" * 70)
    print("TAHAP 2: PENYEIMBANGAN DATA DINAMIS (Bab 3.5)")
    print("=" * 70)

    # Analisis Distribusi Awal 
    print("\n[SEBELUM PENYEIMBANGAN]")
    distribusi = hitung_distribusi_kelas(df)

    # Identifikasi Kelas Minoritas dan Mayoritas
    # Menentukan kelas mana yang memiliki jumlah sampel lebih sedikit (minoritas)
    # dan lebih banyak (mayoritas) secara otomatis.
    kelas_minoritas = min(distribusi, key=distribusi.get)
    kelas_mayoritas = max(distribusi, key=distribusi.get)
    jumlah_minoritas = distribusi[kelas_minoritas]
    jumlah_mayoritas = distribusi[kelas_mayoritas]

    nama_minoritas = "Phishing" if kelas_minoritas == 0 else "Legitimate"
    nama_mayoritas = "Phishing" if kelas_mayoritas == 0 else "Legitimate"

    print(f"\n[INFO] Kelas minoritas : {kelas_minoritas} ({nama_minoritas}) "
          f"= {jumlah_minoritas:,} sampel")
    print(f"[INFO] Kelas mayoritas : {kelas_mayoritas} ({nama_mayoritas}) "
          f"= {jumlah_mayoritas:,} sampel")
    print(f"[INFO] Rasio awal      : 1 : {jumlah_mayoritas / jumlah_minoritas:.2f}")

    # Random Undersampling
    # Mengambil sampel acak dari kelas mayoritas sebanyak jumlah kelas minoritas.
    # Parameter random_state=42 digunakan untuk memastikan reproduksibilitas hasil.
    print(f"\n[PROSES] Melakukan Random Undersampling pada kelas mayoritas...")
    print(f"  Target: {jumlah_mayoritas:,} -> {jumlah_minoritas:,} sampel")

    df_minoritas = df[df['ClassLabel'] == kelas_minoritas]
    df_mayoritas = df[df['ClassLabel'] == kelas_mayoritas]

    # Sampling acak pada kelas mayoritas agar jumlahnya sama dengan kelas minoritas
    df_mayoritas_undersampled = df_mayoritas.sample(
        n=jumlah_minoritas,
        random_state=42
    )

    print(f"  [v] Kelas mayoritas berhasil di-undersample: "
          f"{jumlah_mayoritas:,} -> {len(df_mayoritas_undersampled):,}")

    # Gabungkan Kedua Kelas
    df_seimbang = pd.concat([df_minoritas, df_mayoritas_undersampled], axis=0)

    print(f"  [v] Dataset digabungkan: {len(df_seimbang):,} total sampel")

    return df_seimbang


def shuffle_dataset(df):
    """
    Mengacak urutan baris dalam DataFrame secara acak (shuffle).

    Tujuan: Menghindari bias urutan data saat proses pelatihan model,
    di mana data tidak terkelompok berdasarkan kelas tertentu.

    Parameter:
        df (pd.DataFrame): DataFrame yang akan diacak.

    Returns:
        pd.DataFrame: DataFrame dengan urutan baris yang telah diacak.
    """
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("  [v] Dataset berhasil di-shuffle (diacak urutannya).")
    return df


def simpan_dataset_seimbang(df, path=OUTPUT_SEIMBANG):
    """
    Menyimpan dataset yang telah seimbang ke file CSV untuk keperluan
    dokumentasi dan audit.

    Parameter:
        df (pd.DataFrame): DataFrame yang telah seimbang dan di-shuffle.
        path (str): Lokasi penyimpanan file CSV output.
    """
    # Pastikan direktori output sudah ada
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)
    print(f"  [v] Dataset seimbang disimpan ke: {path}")


def buat_visualisasi_balancing(df_awal, df_seimbang):
    """
    Membuat Grouped Bar Chart distribusi kelas sebelum dan sesudah balancing.
    """
    print("\n[VISUALISASI] Membuat grafik perbandingan sebaran kelas...")
    
    os.makedirs(os.path.join(BASE_DIR, "OUTPUT"), exist_ok=True)
    path_gambar = os.path.join(BASE_DIR, "OUTPUT", "grafik_balancing.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Phishing (0)', 'Legitimate (1)']
    
    # Hitung distribusi
    awal_counts = [len(df_awal[df_awal['ClassLabel'] == 0]), len(df_awal[df_awal['ClassLabel'] == 1])]
    akhir_counts = [len(df_seimbang[df_seimbang['ClassLabel'] == 0]), len(df_seimbang[df_seimbang['ClassLabel'] == 1])]
    
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, awal_counts, width, label='Sebelum Balancing', color='#1f77b4', edgecolor='white')
    rects2 = ax.bar(x + width/2, akhir_counts, width, label='Sesudah Balancing', color='#ff7f0e', edgecolor='white')

    ax.set_ylabel('Jumlah Sampel', fontsize=12)
    ax.set_title('Distribusi Kelas Sebelum vs Sesudah Balancing', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Label angka
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    fig.savefig(path_gambar, dpi=150, facecolor='white')
    plt.close(fig)
    print(f"  [v] Visualisasi disimpan ke: {path_gambar}")


def jalankan_balancing(df):
    """
    Fungsi utama yang mengorkestrasi seluruh proses penyeimbangan data.
    Memanggil fungsi-fungsi di atas secara berurutan.

    Parameter:
        df (pd.DataFrame): DataFrame hasil preprocessing (Tahap 1).

    Returns:
        pd.DataFrame: DataFrame dengan distribusi kelas seimbang (50:50).
    """
    # Lakukan Random Undersampling
    df_seimbang = random_undersampling(df)

    # Shuffle dataset
    df_seimbang = shuffle_dataset(df_seimbang)

    # Simpan ke CSV
    simpan_dataset_seimbang(df_seimbang)

    # Buat visualisasi
    buat_visualisasi_balancing(df, df_seimbang)

    # Tampilkan distribusi akhir
    print("\n[SESUDAH PENYEIMBANGAN]")
    distribusi_akhir = hitung_distribusi_kelas(df_seimbang)

    # Verifikasi bahwa rasio sudah tepat 50:50
    values = list(distribusi_akhir.values())
    if values[0] == values[1]:
        print(f"\n  [v] Rasio kelas berhasil diseimbangkan menjadi 50:50")
    else:
        print(f"\n  [!] PERINGATAN: Rasio belum tepat 50:50!")

    print(f"\n[v] Tahap 2: Penyeimbangan Data selesai.\n")
    return df_seimbang


if __name__ == "__main__":
   
    import importlib.util

    modul_path = os.path.join(BASE_DIR, "01_preprocessing.py")
    spec = importlib.util.spec_from_file_location("preprocessing", modul_path)
    mod_preprocessing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_preprocessing)

    df_bersih = mod_preprocessing.jalankan_preprocessing()
    df_seimbang = jalankan_balancing(df_bersih)
    print(df_seimbang.head(10))
